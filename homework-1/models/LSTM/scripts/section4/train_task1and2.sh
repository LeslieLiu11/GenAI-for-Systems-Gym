#!/bin/bash
#
# train_task1_single_inline.sh
#
# 用於 Section IV, Task I：RNN with Attention。
# 一次提交一個指定的 sequence_length。
#
# 用法：
#   ./train_task1_single_inline.sh <sequence_length>
#   (例如: ./train_task1_single_inline.sh 120)
#
# 會自動呼叫 bsub <<EOF ... EOF，把 #BSUB 行 + python 命令送到 HPC queue。
# 你可以針對不同 seq 多次執行此腳本，一個一個提交。

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sequence_length>"
  exit 1
fi

seq=$1

# 你可依需求調整 total_steps, logs 目錄, trace 檔案位置等
TOTAL_STEPS=30000
EXPERIMENT_BASE_DIR="/share/csc591s25/tliu33/tmp/"
EXPERIMENT_NAME="section4i_rnn_att_seq${seq}"
TRAIN_FILE="/share/csc591s25/traces/astar_313B_train.csv"
VALID_FILE="/share/csc591s25/traces/astar_313B_valid.csv"
LOGS_DIR="/share/csc591s25/models/RNN_with_Attention/logs4"

# 使用 bsub <<EOF 的方式直接提交作業給 HPC
bsub <<EOF
#!/bin/bash
#BSUB -n 1
#BSUB -W 48:00
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -o ${LOGS_DIR}/out_seq${seq}.%J
#BSUB -e ${LOGS_DIR}/err_seq${seq}.%J

module load cuda/11.2
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate /share/csc591s25/conda_env/new_env

cd /share/csc591s25/models/RNN_with_Attention/

python3 -m cache_replacement.policy_learning.cache_model.main \\
  --experiment_base_dir=${EXPERIMENT_BASE_DIR} \\
  --experiment_name=${EXPERIMENT_NAME} \\
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \\
  --model_bindings="loss=[\\"ndcg\\", \\"reuse_dist\\"]" \\
  --model_bindings="address_embedder.max_vocab_size=5000" \\
  --model_bindings="sequence_length=${seq}" \\
  --train_memtrace=${TRAIN_FILE} \\
  --valid_memtrace=${VALID_FILE} \\
  --total_steps=${TOTAL_STEPS}
EOF
