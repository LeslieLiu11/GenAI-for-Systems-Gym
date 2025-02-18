#!/bin/bash
# eval_task6_single.sh
#
# 用法：
#   ./eval_task6_single.sh <neurons> <layers> <act> <dropout> <pruning> <checkpoint_step> > eval_job.sh
#   bsub < eval_job.sh
#
# 說明：
#   1) 類似你在 task5 時的做法，用多行 echo 把最終要提交給 HPC 的腳本輸出到 stdout
#   2) 其中包含 #BSUB 指令 + python3 -m ... 多行參數
#   3) 你再將其重定向到 eval_job.sh，最後 bsub < eval_job.sh 提交
#
# 若 HPC 依舊把 "/share" 變成 "\/share"，可以嘗試調整跳脫層數，或改用 "字串變數 + eval" 等方式。

if [ $# -lt 6 ]; then
  echo "Usage: $0 <neurons> <layers> <act> <dropout> <pruning> <checkpoint_step> > eval_job.sh"
  exit 1
fi

neurons=$1
layers=$2
act=$3
dropout=$4
pruning=$5
checkpoint_step=$6

# experiment_name 與 train_task6_explore.sh 相同
experiment_name="mlp_explore_neu${neurons}_lay${layers}_act${act}_drop${dropout}_prun${pruning}"
eval_experiment_name="eval_${experiment_name}"

TRAIN_BASE_DIR="/share/csc591s25/tliu33/tmp/explore"
EVAL_BASE_DIR="/share/csc591s25/tliu33/tmp/eval_explore"
TEST_TRACE="cache_replacement/policy_learning/cache/traces/astar_313B_test.csv"

checkpoint_file="${checkpoint_step}000.ckpt"
CHECKPOINT_PATH="${TRAIN_BASE_DIR}/${experiment_name}/checkpoints/${checkpoint_file}"
MODEL_CONFIG_PATH="${TRAIN_BASE_DIR}/${experiment_name}/model_config.json"


# 開始逐行 echo，輸出給 HPC 的腳本內容
# 你可以把 #BSUB 指令放在最前面，確保 HPC 能讀取到 queue、gpu 等參數

echo "#!/bin/bash"
echo ""
echo "#BSUB -n 1"
echo "#BSUB -W 71:59"
echo "#BSUB -o /share/csc591s25/tliu33/tmp/logs/out.${bm}.eval.%J"
echo "#BSUB -e /share/csc591s25/tliu33/tmp/logs/err.${bm}.eval.%J"
echo "#BSUB -J hw5_${bm}_${pol}_lr${lr}"
echo ""
echo "source ~/.bashrc"
echo "conda activate /share/csc591s25/conda_env/new_env"
echo "cd /share/csc591s25/models/MLP"

# 多行 echo python3 -m ...
# 你可以在 --config_bindings= 參數裡酌情加/減反斜線
echo "python3 -m cache_replacement.policy_learning.cache.main \\"
echo "  --experiment_base_dir=\"${EVAL_BASE_DIR}\" \\"
echo "  --experiment_name=\"${eval_experiment_name}\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json\" \\"
echo "  --memtrace_file=\"${TEST_TRACE}\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.checkpoint=\\\"${CHECKPOINT_PATH}\\\"\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.config_path=\\\"${MODEL_CONFIG_PATH}\\\"\" \\"
echo "  --warmup_period=0"

