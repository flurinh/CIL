#!/bin/bash

declare -a LearningRates=("0.0001")
declare -a BatchSizes=("1")
declare -a Optimizers=("2")
declare -a TrainingSet=("2" "4")
# Iterate the string array using for loop
for lr in ${LearningRates[@]}; do
   for bs in ${BatchSizes[@]}; do
        for opt in ${Optimizers[@]}; do
            for ts in ${TrainingSet[@]}; do
              name="lr_${lr}_bs_${bs}_opt_${opt}_R2AttU_set${ts}"
              bsub -n 4 -W 25:00 -R "rusage[mem=4096, ngpus_excl_p=2]" python train.py --lr $lr -b $bs -e 50 --op $opt -d $ts --log $name
              sleep 1
              done
        done
   done
done