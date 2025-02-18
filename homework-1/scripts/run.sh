#!/bin/bash
#BSUB -n 1
#BSUB -W 48:00
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -J training_phase
#BSUB -o /share/csc591s25/aghanba2/GenAI-for-Systems-Gym/homework-1/models/LSTM/logs/out.%J
#BSUB -e /share/csc591s25/aghanba2/GenAI-for-Systems-Gym/homework-1/models/LSTM/logs/err.%J



source ~/.bashrc
conda activate /share/$GROUP/conda_env/new_env
cd /share/$GROUP/$USER/GenAI-for-Systems-Gym/homework-1/models/LSTM



python3 -m cache_replacement.policy_learning.cache.main \
--experiment_base_dir=/share/$GROUP/$USER/GenAI-for-Systems-Gym/homework-1/models/LSTM/experiments \
--experiment_name=astar_belady \
--cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
--cache_configs=cache_replacement/policy_learning/cache/configs/eviction_policy/belady.json \
--memtrace_file=/share/$GROUP/traces/astar_313B_test.csv