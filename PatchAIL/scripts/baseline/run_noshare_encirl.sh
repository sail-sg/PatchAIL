#!/bin/bash

# EncIRL
for i in {4..5}
do
    python train.py agent=noshare_encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=noshare_encgail reward_type=gail
    python train.py agent=noshare_encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=noshare_encairl reward_type=airl
    python train.py agent=noshare_encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=${i} algo_name=noshare_encgail2 reward_type=gail2
done
python train.py agent=noshare_encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=3 algo_name=noshare_encgail reward_type=gail2
python train.py agent=noshare_encirl suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=$2 seed=3 algo_name=noshare_encgail reward_type=airl
