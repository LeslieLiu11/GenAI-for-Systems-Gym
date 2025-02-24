#!/bin/bash
#
# 用於 Section IV, Task I：RNN with Attention，一次提交一個指定的 sequence_length。
# 執行方式：
#   ./train_task1_single_inline.sh <sequence_length>
# 會自動產生一個 HPC 作業檔 (job_<seq>.sh)，再用 bsub 提交給 GPU queue。
#
# 若仍被分配到 debug queue，請確定 HPC 系統允許使用 "gpu" queue，且
# 你的帳號有權限在該 queue 執行。

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sequence_length>"
  exit 1
fi

seq=$1

# 你可依需求調整以下參數
TOTAL_STEPS=30000
EXPERIMENT_BASE_DIR="/share/csc591s25/tliu33/tmp_v2/"
EXPERIMENT_NAME="section4_rnn_wo_att_seq${seq}"
TRAIN_FILE="/share/csc591s25/tliu33/models/MLP/cache_replacement/policy_learning/cache/traces/astar_313B_train.csv"
VALID_FILE="/share/csc591s25/tliu33/models/MLP/cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv"
LOGS_DIR="/share/csc591s25/tliu33/log_v2"
MODEL_DIR="/share/csc591s25/tliu33/models/RNN_without_Attention/"  # 你的程式碼目錄

# 產生 HPC job 檔案名稱
JOBFILE="job_${seq}.sh"

cat <<EOF > "${JOBFILE}"
#!/bin/bash
#BSUB -n 1
#BSUB -W 48:00
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -o ${LOGS_DIR}/out_seq${seq}.%J
#BSUB -e ${LOGS_DIR}/err_seq${seq}.%J

# 啟用調試模式
set -x

# 載入環境
module load cuda/11.2
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate /share/csc591s25/conda_env/new_env

# 切換到程式碼目錄
cd "${MODEL_DIR}"

# 執行命令
python3 -m cache_replacement.policy_learning.cache_model.main \\
  --experiment_base_dir="${EXPERIMENT_BASE_DIR}" \\
  --experiment_name="${EXPERIMENT_NAME}" \\
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \\
  --model_bindings="loss=[\\"ndcg\\", \\"reuse_dist\\"]" \\
  --model_bindings="address_embedder.max_vocab_size=5000" \\
  --model_bindings="sequence_length=${seq}" \\
  --train_memtrace="${TRAIN_FILE}" \\
  --valid_memtrace="${VALID_FILE}" \\
  --total_steps=${TOTAL_STEPS}
EOF

# 讓 job 檔具可執行權限（不是必要，但習慣上安全）
chmod +x "${JOBFILE}"

echo "已產生 HPC 作業檔案: ${JOBFILE}"
echo "現在提交 bsub < ${JOBFILE}..."
bsub < "${JOBFILE}"
