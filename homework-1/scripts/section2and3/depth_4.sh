#!/bin/bash
#BSUB -n 1
#BSUB -W 12:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"

# 這裡請直接寫死路徑，或在 #BSUB 之前定義
# (LSF 不會幫你展開在 #BSUB 行中設定的變數)
# 如果要自訂輸出路徑，就硬寫:
#BSUB -o /share/csc591s25/tliu33/log_v2/mlp_training.out.%J
#BSUB -e /share/csc591s25/tliu33/log_v2/mlp_training.err.%J

#----------------------------------------
set -x

module load PrgEnv-pgi
module load cuda/11.2
export LD_LIBRARY_PATH=/usr/local/apps/cuda/cuda-11.2/lib64:$LD_LIBRARY_PATH

# 若你想用 .bashrc，就確保 conda init 正常
source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

nvidia-smi

# 若目錄名稱有空白，請加引號，或改目錄名
cd "/share/csc591s25/tliu33/models/MLP"

# 確認是 cache_model.main，而不是 cache_model1
python3 -m cache_replacement.policy_learning.cache_model4.main \
  --experiment_base_dir=/share/csc591s25/tliu33/tmp_v2 \
  --experiment_name=mlp_width_128_d4 \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="lstm_hidden_size=128" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/astar_313B_train.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv
