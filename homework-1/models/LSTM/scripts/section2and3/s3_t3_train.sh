#!/bin/bash
# 此腳本用於探索 MLP 模型的激活函數對 cache hit rate 的影響。
# 變化的激活函數包括：ReLU, LeakyReLU, sigmoid, tanh。
# 固定每層神經元數 128，層數使用預設值（例如默認為 2 層，視你的模型實作而定）。
#
# Usage: ./task3.sh <benchmark>
# 例如: ./task3.sh astar

if [ $# -lt 1 ]; then
    echo "Usage: $0 <benchmark>"
    exit 1
fi

bm=$1
# 定義要測試的激活函數列表
activations=("ReLU" "LeakyReLU" "sigmoid" "tanh")

for act in "${activations[@]}"; do
    experiment_name="${bm}_act_${act}"
    echo "Submitting experiment: ${experiment_name} with activation function ${act}..."
    bsub <<EOF
#!/bin/bash
#BSUB -n 1
#BSUB -W 12:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o logs/out.${experiment_name}.%J
#BSUB -e logs/err.${experiment_name}.%J
#BSUB -J hw1_${bm}_act_${act}

# 載入環境設定
source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

# 切換到 MLP 模型原始碼目錄
cd /share/csc591s25/models/MLP

# 執行訓練命令，固定每層 128 個神經元，並設定激活函數
python3 -m cache_replacement.policy_learning.cache_model.main \\
  --experiment_base_dir="/tmp" \\
  --experiment_name="${experiment_name}" \\
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \\
  --model_bindings="lstm_hidden_size=128" \\
  --model_bindings="activation="${act}"" \\
  --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" \\
  --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv"
EOF
done
