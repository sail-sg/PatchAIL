#!/bin/bash

# EncIRL
for i in {1..5}
do
    python train.py agent=encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=encgail reward_type=gail reward_scale=5.0
    python train.py agent=encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=encairl reward_type=airl reward_scale=5.0
    python train.py agent=encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=encgail2 reward_type=gail2 reward_scale=5.0
done
