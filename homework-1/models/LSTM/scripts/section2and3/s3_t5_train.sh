#!/bin/bash
# This script submits a training job for each learning rate variant.
# The learning rates tested are: 1e-5, 1e-3 (default), and 1e-1.
# The model uses default settings for other parameters (e.g., 128 neurons per layer).
# Training logs (including epoch-by-epoch normalized cache hit rate) are stored in the TensorBoard directory.
#
# Usage: ./train_task5.sh <benchmark>
# Example: ./train_task5.sh astar

if [ $# -lt 1 ]; then
    echo "Usage: $0 <benchmark>"
    exit 1
fi

bm=$1
# Define the learning rates to test
learning_rates=("1e-5" "1e-3" "1e-1")

for lr in "${learning_rates[@]}"; do
    experiment_name="${bm}_lr_${lr}"
    echo "Submitting experiment: ${experiment_name} with learning rate ${lr}..."
    bsub <<EOF
#!/bin/bash
#BSUB -n 1                           # Request 1 CPU core
#BSUB -W 12:00                       # Maximum wall time: 12 hours
#BSUB -q gpu                         # Use the GPU queue
#BSUB -gpu "num=1:mode=shared:mps=no"  # Request 1 GPU (shared mode)
#BSUB -o ${experiment_name}.out.%J    # Standard output file
#BSUB -e ${experiment_name}.err.%J    # Standard error file
#BSUB -J ${experiment_name}

# Enable debug mode for tracing
set -x

# Load necessary modules (adjust versions as available)
module load PrgEnv-pgi              # Load PGI environment
module load cuda/11.2               # Example: using CUDA 11.2
export LD_LIBRARY_PATH=/usr/local/apps/cuda/cuda-11.2/lib64:\$LD_LIBRARY_PATH

# Initialize conda and activate the virtual environment
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate /share/csc591s25/conda_env/new_env

# (Optional) Check GPU status
nvidia-smi

# Change to the directory containing the MLP model code
cd /share/csc591s25/models/MLP

# Execute the training command with the specified learning rate.
python3 -m cache_replacement.policy_learning.cache_model.main \\
  --experiment_base_dir="/share/csc591s25/tliu33/tmp" \\
  --experiment_name="${experiment_name}" \\
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \\
  --model_bindings="lstm_hidden_size=128" \\
  --model_bindings="learning_rate=${lr}" \\
  --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" \\
  --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv"
EOF
done
