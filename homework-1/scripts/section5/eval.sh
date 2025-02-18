#!/bin/bash
#BSUB -n 1                            # 申請 1 個 CPU core
#BSUB -W 12:00                        # 最長執行時間 12 小時
#BSUB -q gpu                          # 使用 GPU queue
#BSUB -gpu "num=1:mode=shared:mps=no" # 申請 1 顆 GPU (共享模式)
#BSUB -o /share/csc591s25/tliu33/tmp/logs/lstm_eval.out.%J  # 標準輸出日誌
#BSUB -e /share/csc591s25/tliu33/tmp/logs/lstm_eval.err.%J  # 錯誤日誌
#BSUB -J lstm_eval                    # 作業名稱

#------------------------------------------------------------
# 啟用調試模式，顯示每個命令的執行情況
set -x

#------------------------------------------------------------
# 載入必要的模組 (請依照 HPC 實際可用的模組)
module load PrgEnv-pgi
module load cuda/11.2
export LD_LIBRARY_PATH=/usr/local/apps/cuda/cuda-11.2/lib64:$LD_LIBRARY_PATH

#------------------------------------------------------------
# 初始化 conda 並啟用虛擬環境
source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

# (可選) 檢查 GPU 狀態
nvidia-smi

#------------------------------------------------------------
# 切換到 LSTM 模型原始碼所在目錄
cd /share/csc591s25/tliu33/LSTM

#------------------------------------------------------------
# 執行「模擬 / 評估」指令 (cache.main)
# 注意：要使用 learned eviction policy (learned.json)，
#       並透過 config_bindings 指定 checkpoint 與 model_config。
CHECKPOINT_PATH="/share/csc591s25/tliu33/tmp/section5/section5_ablat_22/checkpoints/20000.ckpt"
MODEL_CONFIG="/share/csc591s25/tliu33/tmp/section5/section5_ablat_22/model_config.json"

python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir="/share/csc591s25/tliu33/tmp/eval_section5" \
  --experiment_name="eval_section5_ablat_22" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="cache_replacement/policy_learning/cache/traces/astar_test.csv" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"${CHECKPOINT_PATH}\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"${MODEL_CONFIG}\"" \
  --warmup_period=0
