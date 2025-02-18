#!/bin/bash
# This script submits a training job for each MLP depth: 1, 2, 3, and 4.
# The model uses 128 neurons per layer (fixed), and the number of layers is varied.

# List of desired model depths
for depth in 1 2 3 4; do
    experiment_name="mlp_depth_${depth}"
    echo "Submitting experiment: ${experiment_name} with ${depth} layers..."
    bsub <<EOF
#!/bin/bash
#BSUB -n 1                           # Request 1 CPU core
#BSUB -W 12:00                       # Maximum wall time: 12 hours
#BSUB -q gpu                         # Use the GPU queue
#BSUB -gpu "num=1:mode=shared:mps=no"  # Request 1 GPU (shared mode)
#BSUB -o ${experiment_name}.out.%J    # Standard output file
#BSUB -e ${experiment_name}.err.%J    # Standard error file

# Enable debug mode
set -x

# Load necessary modules (adjust versions as available on your system)
module load PrgEnv-pgi              # Load PGI environment
module load cuda/11.2               # For example, using CUDA 11.2
export LD_LIBRARY_PATH=/usr/local/apps/cuda/cuda-11.2/lib64:\$LD_LIBRARY_PATH

# Initialize conda and activate the virtual environment
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate /share/csc591s25/conda_env/new_env

# (Optional) Check GPU status
nvidia-smi

# Change to the directory containing the MLP model code
cd /share/csc591s25/models/MLP

# Execute the training command
python3 -m cache_replacement.policy_learning.cache_model.main \\
  --experiment_base_dir=/tmp \\
  --experiment_name=${experiment_name} \\
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \\
  --model_bindings="lstm_hidden_size=128" \\
  --model_bindings="num_layers=${depth}" \\
  --train_memtrace=cache_replacement/policy_learning/cache/traces/astar_313B_train.csv \\
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv
EOF
done
