#!/bin/bash

# BC
for i in {1..5}
do
    python train.py agent=bc suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=10 seed=${i} algo_name=bc
    python train.py agent=bc suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=5 seed=${i} algo_name=bc
    python train.py agent=bc suite=dmc obs_type=pixels suite/dmc_task=$1 num_demos=1 seed=${i} algo_name=bc
done