#!/bin/bash
# This script demonstrates how to explore multiple hyperparameters for MLP training.
# Example hyperparams: neurons, layers, activation, dropout, pruning, etc.
#
# Usage: ./train_task6_explore.sh
#
# NOTE:
#   1) We have fixed the quoting issue for activation and other string parameters.
#   2) If your code does not support pruning or dropout, please remove or comment them out.

# Arrays for each hyperparam you want to explore
NEURONS_ARR=(128)
LAYERS_ARR=(1 2)
ACT_ARR=("ReLU" "sigmoid")     # Activation functions
DROPOUT_ARR=(0.0 0.2)          # If your code supports dropout
PRUNING_ARR=("False" "True")   # If your code supports pruning

for neurons in "${NEURONS_ARR[@]}"; do
  for layers in "${LAYERS_ARR[@]}"; do
    for act in "${ACT_ARR[@]}"; do
      for dropout in "${DROPOUT_ARR[@]}"; do
        for pruning in "${PRUNING_ARR[@]}"; do

          # Construct an experiment_name to differentiate runs
          experiment_name="mlp_explore_neu${neurons}_lay${layers}_act${act}_drop${dropout}_prun${pruning}"

          # Submit BSUB job
          bsub <<EOF
#!/bin/bash
#BSUB -n 1
#BSUB -W 12:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o train_explore.out.%J
#BSUB -e train_explore.err.%J
#BSUB -J ${experiment_name}


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

# Move to MLP code directory
cd /share/csc591s25/models/MLP

python3 -m cache_replacement.policy_learning.cache_model.main \\
  --experiment_base_dir="/share/csc591s25/tliu33/tmp/explore" \\
  --experiment_name="${experiment_name}" \\
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \\
  --model_bindings="lstm_hidden_size=${neurons}" \\
  --model_bindings="num_layers=${layers}" \\
  --model_bindings="activation=\"${act}\"" \\
  --model_bindings="dropout=${dropout}" \\
  --model_bindings="pruning=${pruning}" \\
  --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" \\
  --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv"
EOF

        done
      done
    done
  done
done