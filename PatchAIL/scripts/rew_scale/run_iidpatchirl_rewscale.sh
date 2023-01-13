#!/bin/bash

# PatchIRL
for i in {2..5}
do
    python train.py agent=iidpatchirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=iidpatchairl_ss reward_type=airl reward_scale=10.0
    python train.py agent=iidpatchirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=iidpatchgail_ss reward_type=gail reward_scale=10.0
    python train.py agent=iidpatchirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=iidpatchgail2_ss reward_type=gail2 reward_scale=10.0
done 
