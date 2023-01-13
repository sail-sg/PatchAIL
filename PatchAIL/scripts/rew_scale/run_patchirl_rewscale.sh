#!/bin/bash

# PatchIRL
for i in {1..5}
do
    # python train.py agent=patchirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=patchgail_ss reward_type=gail reward_scale=5.0
    python train.py agent=patchirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=patchairl_ss reward_type=airl reward_scale=10.0
    # python train.py agent=patchirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=patchgail2_ss reward_type=gail2 reward_scale=5.0
done