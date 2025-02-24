#!/bin/bash
#
# eval_task1_single_echo.sh
#
# 用於：一次只評估一個指定 sequence_length 的 RNN_without_Attention 模型。
#       使用 learned eviction policy (checkpoint=20000.ckpt) 來在 test trace 上測試 hit rate。
#
# 用法：
#   ./eval_task1_single_echo.sh <sequence_length> > jobXX.sh
#   bsub < jobXX.sh
#
#   例如：
#   ./eval_task1_single_echo.sh 120 > job120.sh
#   bsub < job120.sh
#
# 這樣可以避免在本地執行 Python，也可避免 "\/share" 的意外轉義。

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sequence_length> > jobXX.sh"
  echo "Then do: bsub < jobXX.sh"
  exit 1
fi

seq=$1

# 假設在訓練時，experiment_name = "section4i_rnn_woatt_seq${seq}"，跑到 20000 steps
CHECKPOINT_STEP=20000

# HPC 上的 base dir (與 train 相同)
EXPERIMENT_BASE_DIR="/share/csc591s25/tliu33/tmp_v2"
TRAIN_EXPERIMENT_NAME="section4_rnn_wo_att_seq${seq}"

CHECKPOINT_FILE="${CHECKPOINT_STEP}.ckpt"
CHECKPOINT_PATH="${EXPERIMENT_BASE_DIR}/${TRAIN_EXPERIMENT_NAME}/checkpoints/${CHECKPOINT_FILE}"
MODEL_CONFIG_PATH="${EXPERIMENT_BASE_DIR}/${TRAIN_EXPERIMENT_NAME}/model_config.json"

# 新的 eval experiment_name
EVAL_EXPERIMENT_NAME="eval_${TRAIN_EXPERIMENT_NAME}"

# 測試檔案
TEST_FILE="/share/csc591s25/tliu33/models/MLP/cache_replacement/policy_learning/cache/traces/astar_313B_test.csv"

# HPC log 輸出路徑
LOGS_DIR="/share/csc591s25/tliu33/log_v2"

# 以下用 echo 一行一行輸出 HPC 作業腳本內容
echo "#!/bin/bash"
echo "#BSUB -n 1"
echo "#BSUB -W 48:00"
echo "#BSUB -o ${LOGS_DIR}/out_eval_seq${seq}.%J"
echo "#BSUB -e ${LOGS_DIR}/err_eval_seq${seq}.%J"
echo "#BSUB -J hw1_att_wo_${seq}"
echo ""
echo "source ~/.bashrc"
echo "conda activate /share/csc591s25/conda_env/new_env"

# 切換到 RNN_without_Attention 目錄
echo "cd /share/csc591s25/tliu33/models/RNN_without_Attention/"

# 執行 learned eviction policy 做測試
echo "python3 -m cache_replacement.policy_learning.cache.main \\"
echo "  --experiment_base_dir=\"${EXPERIMENT_BASE_DIR}\" \\"
echo "  --experiment_name=\"${EVAL_EXPERIMENT_NAME}\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json\" \\"
echo "  --memtrace_file=\"${TEST_FILE}\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.checkpoint=\\\"${CHECKPOINT_PATH}\\\"\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.config_path=\\\"${MODEL_CONFIG_PATH}\\\"\" \\"
echo "  --warmup_period=0"


