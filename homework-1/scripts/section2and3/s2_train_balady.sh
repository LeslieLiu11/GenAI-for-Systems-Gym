python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir=/share/csc591s25/tliu33/tmp \
  --experiment_name=sample_belady_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --cache_configs=cache_replacement/policy_learning/cache/configs/eviction_policy/lru.json \
  --memtrace_file=cache_replacement/policy_learning/cache/traces/astar_313B_test.csv

