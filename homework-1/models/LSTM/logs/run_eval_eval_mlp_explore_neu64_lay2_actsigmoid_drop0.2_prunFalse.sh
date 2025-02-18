#!/bin/bash
#BSUB -n 1
#BSUB -W 12:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o logs/out.eval_mlp_explore_neu64_lay2_actsigmoid_drop0.2_prunFalse.%J
#BSUB -e logs/err.eval_mlp_explore_neu64_lay2_actsigmoid_drop0.2_prunFalse.%J
#BSUB -J eval_mlp_explore_neu64_lay2_actsigmoid_drop0.2_prunFalse

set -x
module load PrgEnv-pgi
module load cuda/11.2
export LD_LIBRARY_PATH=/usr/local/apps/cuda/cuda-11.2/lib64:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /share/csc591s25/conda_env/new_env

nvidia-smi
cd /share/csc591s25/models/MLP

# 這裡改用「字串變數 + eval」單行命令，避免 HPC 對多行進行轉義
CMD="python3 -m cache_replacement.policy_learning.cache.main   --experiment_base_dir=\"/share/csc591s25/tliu33/tmp/eval_explore\"   --experiment_name=\"eval_mlp_explore_neu64_lay2_actsigmoid_drop0.2_prunFalse\"   --cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\"   --cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json\"   --memtrace_file=\"cache_replacement/policy_learning/cache/traces/astar_313B_test.csv\"   --config_bindings=\"eviction_policy.scorer.checkpoint=\\\\"/share/csc591s25/tliu33/tmp/explore/mlp_explore_neu64_lay2_actsigmoid_drop0.2_prunFalse/checkpoints/20000.ckpt\\\\"\"   --config_bindings=\"eviction_policy.scorer.config_path=\\\\"/share/csc591s25/tliu33/tmp/explore/mlp_explore_neu64_lay2_actsigmoid_drop0.2_prunFalse/model_config.json\\\\"\"   --warmup_period=0
"

echo "Will run: $CMD"
eval "$CMD"
