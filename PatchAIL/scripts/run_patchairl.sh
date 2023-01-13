#!/bin/bash

# PatchIRL


CUDA_VISIBLE_DEVICES=0 python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=2 algo_name=patchairl_ss_kldsimregauto1.3_buf100w_randomexpdemo_noaugsim reward_type=airl &
CUDA_VISIBLE_DEVICES=1 python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=3 algo_name=patchairl_ss_kldsimregauto1.3_buf100w_randomexpdemo_noaugsim reward_type=airl &
CUDA_VISIBLE_DEVICES=0 python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=4 algo_name=patchairl_ss_kldsimregauto1.3_buf100w_randomexpdemo_noaugsim reward_type=airl &
CUDA_VISIBLE_DEVICES=1 python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=1 algo_name=patchairl_ss_kldsimregauto1.3_buf100w_randomexpdemo_noaugsim reward_type=airl &
CUDA_VISIBLE_DEVICES=0 python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=5 algo_name=patchairl_ss_kldsimregauto1.3_buf100w_randomexpdemo_noaugsim reward_type=airl 

# python train.py agent=patchirl_simreg suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=5 algo_name=patchairl_ss_kldsimreg1.5_buf100w reward_type=airl
