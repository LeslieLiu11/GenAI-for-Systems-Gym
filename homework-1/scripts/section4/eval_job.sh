#!/bin/bash
#BSUB -n 1
#BSUB -W 48:00
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -o /share/csc591s25/tliu33/tmp/logs_eval_att/out_eval_seq120.%J
#BSUB -e /share/csc591s25/tliu33/tmp/logs_eval_att/err_eval_seq120.%J
#BSUB -J hw1_att_120

source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env
cd /share/csc591s25/models/RNN_with_Attention/
python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir="/share/csc591s25/tliu33/tmp" \
  --experiment_name="eval_section4i_rnn_att_seq120" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="/share/csc591s25/traces/astar_313B_test.csv" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"/share/csc591s25/tliu33/tmp/section4i_rnn_att_seq120/checkpoints/20000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"/share/csc591s25/tliu33/tmp/section4i_rnn_att_seq120/model_config.json\"" \
  --warmup_period=0
