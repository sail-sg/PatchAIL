#!/bin/bash

# PatchIRL
for i in {1..5}
do
    python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=patchgail_ss_kldsimreg1.5_buf100w reward_type=gail &
    python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=patchgail2_ss_kldsimreg1.5_buf100w reward_type=gail2 &
done