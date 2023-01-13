#!/bin/bash

# EncIRL
for i in {1..5}
do
    python train.py agent=encirl_ss suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=encgail_ss reward_type=gail replay_buffer_size=150000
    python train.py agent=encirl_ss suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=encairl_ss reward_type=airl replay_buffer_size=150000
    python train.py agent=encirl_ss suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=encgail2_ss reward_type=gail2 replay_buffer_size=150000
done