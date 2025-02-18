#!/bin/bash
#
# eval_section4_task3_echo_single.sh

#
# 用法：
#   chmod +x eval_section4_task3_echo_single.sh
#   ./eval_section4_task3_echo_single.sh <ah> > jobAH.sh
#   bsub < jobAH.sh


if [ $# -lt 1 ]; then
  echo "Usage: $0 <attention_history_length> > jobAH.sh"
  echo "Then: bsub < jobAH.sh"
  exit 1
fi

ah=$1

# 假設在 train_section4_task3.sh 時，experiment_name = "section4iii_rnn_att_ah${ah}"，跑到 30000 steps
CHECKPOINT_STEP=20000
CHECKPOINT_FILE="${CHECKPOINT_STEP}.ckpt"

# 訓練 base dir & HPC log 路徑
TRAIN_BASE_DIR="/share/csc591s25/tliu33/tmp"
EVAL_BASE_DIR="/share/csc591s25/tliu33/tmp/eval_att_task3"
LOGS_DIR="/share/csc591s25/models/RNN_with_Attention/logs_task3_eval"

TRAIN_EXPERIMENT_NAME="section4_rnn_att_ah${ah}"
CHECKPOINT_PATH="${TRAIN_BASE_DIR}/${TRAIN_EXPERIMENT_NAME}/checkpoints/${CHECKPOINT_FILE}"
MODEL_CONFIG_PATH="${TRAIN_BASE_DIR}/${TRAIN_EXPERIMENT_NAME}/model_config.json"
EVAL_EXPERIMENT_NAME="eval_${TRAIN_EXPERIMENT_NAME}"

# 測試檔
TEST_FILE="/share/csc591s25/traces/astar_313B_test.csv"

# 以下開始「多行 echo」輸出 HPC job script
echo "#!/bin/bash"
echo "#BSUB -n 1"
echo "#BSUB -W 24:00"
echo "#BSUB -q gpu"
echo "#BSUB -gpu \"num=1\""
echo "#BSUB -o ${LOGS_DIR}/out_eval_ah${ah}.%J"
echo "#BSUB -e ${LOGS_DIR}/err_eval_ah${ah}.%J"
echo "#BSUB -J eval_section4iii_ah${ah}"

echo "source ~/.bashrc"
echo "conda activate /share/csc591s25/conda_env/new_env"

echo "cd /share/csc591s25/models/RNN_with_Attention/"

# 執行 learned eviction policy，讀取 checkpoint 做測試
echo "python3 -m cache_replacement.policy_learning.cache.main \\"
echo "  --experiment_base_dir=\"${EVAL_BASE_DIR}\" \\"
echo "  --experiment_name=\"${EVAL_EXPERIMENT_NAME}\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json\" \\"
echo "  --memtrace_file=\"${TEST_FILE}\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.checkpoint=\\\"${CHECKPOINT_PATH}\\\"\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.config_path=\\\"${MODEL_CONFIG_PATH}\\\"\" \\"
echo "  --warmup_period=0"
