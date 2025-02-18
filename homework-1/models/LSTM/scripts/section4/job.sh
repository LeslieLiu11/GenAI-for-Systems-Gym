#!/bin/bash
#BSUB -n 1
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -o /share/csc591s25/models/RNN_with_Attention/logs_task3_eval/out_eval_ah30.%J
#BSUB -e /share/csc591s25/models/RNN_with_Attention/logs_task3_eval/err_eval_ah30.%J
#BSUB -J eval_section4iii_ah30
source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env
cd /share/csc591s25/models/RNN_with_Attention/
python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir="/share/csc591s25/tliu33/tmp/eval_att_task3" \
  --experiment_name="eval_section4iii_rnn_att_ah30" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="/share/csc591s25/traces/astar_313B_test.csv" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"/share/csc591s25/tliu33/tmp/section4iii_rnn_att_ah30/checkpoints/20000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"/share/csc591s25/tliu33/tmp/section4iii_rnn_att_ah30/model_config.json\"" \
  --warmup_period=0
