#!/bin/bash 

if [ $# -lt 5 ]; then
    echo "usage: eval.sh <benchmark> <tracelen> <checkpoint> <policy> <neurons> [opt:testcsv]"
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

neurons=$5

# 訓練時模型檔案存放在 mlp_width_<neurons> 資料夾中
traindir="mlp_width_${neurons}"

if [ $# -eq 6 ]; then
    testcsv=$6
else
    testcsv="${bm}_${len}_test"
fi

echo "#!/bin/bash"
echo ""
echo "#BSUB -n 1"
echo "#BSUB -W 71:59"
echo "#BSUB -o /share/csc591s25/tliu33/tmp/logs/out.${bm}.eval.%J"
echo "#BSUB -e /share/csc591s25/tliu33/tmp/logs/err.${bm}.eval.%J"
echo "#BSUB -J hw1_${bm}_${pol}"
echo ""
echo "source ~/.bashrc"
echo "conda activate /share/csc591s25/conda_env/new_env"
echo "python3 -m cache_replacement.policy_learning.cache.main \\"
echo "  --experiment_base_dir=\"/share/csc591s25/tliu33/tmp/eval\" \\"
echo "  --experiment_name=\"${bm}${len}_chkpt${neurons}k\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/default.json\" \\"
echo "  --cache_configs=\"cache_replacement/policy_learning/cache/configs/eviction_policy/${pol}.json\" \\"
echo "  --memtrace_file=\"/share/csc591s25/models/MLP/cache_replacement/policy_learning/cache/traces/${testcsv}.csv\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.checkpoint=\\\"/share/csc591s25/tliu33/tmp/${traindir}/checkpoints/${chkpt}000.ckpt\\\"\" \\"
echo "  --config_bindings=\"eviction_policy.scorer.config_path=\\\"/share/csc591s25/tliu33/tmp/${traindir}/model_config.json\\\"\" \\"
echo "  --warmup_period=0"

