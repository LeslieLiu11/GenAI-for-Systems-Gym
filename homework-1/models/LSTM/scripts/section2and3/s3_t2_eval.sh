#!/bin/bash

# usage: eval_task2_depth.sh <benchmark> <tracelen> <checkpoint> <policy> <layers> [opt:testcsv]
# 例如：
#   ./eval_task2_depth.sh astar 313B 20 0 1
# 這表示：
#   - benchmark=astar
#   - tracelen=313B
#   - checkpoint=20（即讀取20000.ckpt）
#   - policy=0（learned）
#   - layers=1（對應資料夾 mlp_depth_1）
#   - 預設使用 testcsv = "<benchmark>_<tracelen>_test"

if [ $# -lt 5 ]; then
    echo "Usage: $0 <benchmark> <tracelen> <checkpoint> <policy> <layers> [opt:testcsv]"
    exit 1
fi

bm=$1
len=$2
chkpt=$3

if [ $4 -eq 1 ]; then
    pol="lru"
elif [ $4 -eq 2 ]; then
    pol="belady"
else
    pol="learned"
fi

layers=$5

# 假設訓練時資料夾名稱為 mlp_depth_<layers>
traindir="mlp_depth4_${layers}"

if [ $# -eq 6 ]; then
    testcsv=$6
else
    testcsv="${bm}_${len}_test"
fi

# 輸出 BSUB 作業腳本
echo "#!/bin/bash"
echo ""
echo "#BSUB -n 1"
echo "#BSUB -W 71:59"
echo "#BSUB -o /share/csc591s25/tliu33/tmp/logs/out.${bm}.eval.%J"
echo "#BSUB -e /share/csc591s25/tliu33/tmp/logs/err.${bm}.eval.%J"
echo "#BSUB -J hw2_${bm}_${pol}_layers${layers}"
echo ""
echo "source ~/.bashrc"
echo "conda activate /share/csc591s25/conda_env/new_env"

echo "python3 -m cache_replacement.policy_learning.cache.main \\"
echo "  --experiment_base_dir=\"/share/csc591s25/tliu33/tmp/eval\" \\"
echo "  --experiment_name=\"${bm}${len}_chkpt${layers}layers4\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/${pol}.json\" \\"
echo "  --memtrace_file=\"/share/csc591s25/models/MLP/cache_replacement/policy_learning/cache/traces/${testcsv}.csv\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.checkpoint=\\\"/share/csc591s25/tliu33/tmp/${traindir}/checkpoints/${chkpt}000.ckpt\\\"\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.config_path=\\\"/share/csc591s25/tliu33/tmp/${traindir}/model_config.json\\\"\" \\"
echo "  --warmup_period=0"
