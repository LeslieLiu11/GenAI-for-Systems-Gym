#!/bin/bash
#BSUB -n 1                          # 申請 1 個核心
#BSUB -W 12:00                      # 作業最大執行時間 12 小時
#BSUB -q gpu                        # 使用 GPU queue
#BSUB -gpu "num=1:mode=shared:mps=no" # 申請 1 顆 GPU (共享模式，不使用 MPS)
#BSUB -o /share/csc591s25/tliu33/tmp/logs/lstm_training.out.%J  # 標準輸出日誌
#BSUB -e /share/csc591s25/tliu33/tmp/logs/lstm_training.err.%J  # 錯誤日誌

#------------------------------------------------------------
# 啟用調試模式，顯示每個命令的執行情況
set -x

#------------------------------------------------------------
# 載入必要的模組，選擇 CUDA 11.2（與 cudatoolkit 11.8 搭配較好）
module load PrgEnv-pgi                      # 載入 PGI 編譯環境
module load cuda/11.2                       # 載入 CUDA 模組 (使用 11.2)

# 設定 LD_LIBRARY_PATH 以確保載入正確的 CUDA 庫
export LD_LIBRARY_PATH=/usr/local/apps/cuda/cuda-11.2/lib64:$LD_LIBRARY_PATH

#------------------------------------------------------------
# 初始化 conda 並激活虛擬環境
source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

#------------------------------------------------------------
# (可選) 檢查 GPU 狀態
nvidia-smi

#------------------------------------------------------------
# 切換到 MLP 模型原始碼所在目錄
cd /share/csc591s25/models/LSTM


python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/share/csc591s25/tliu33/tmp/section5 \
  --experiment_name=section5_default2 \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/astar_train.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/astar_valid.csv
