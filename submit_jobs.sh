#!/bin/bash

declare -a LearningRates=("0.001" "0.0005" "0.0001")
declare -a BatchSizes=("1" "2")
declare -a Optimizers=("2")
# Iterate the string array using for loop
for lr in ${LearningRates[@]}; do
   for bs in ${BatchSizes[@]}; do
        for opt in ${Optimizers[@]}; do
              name="lr_${lr}_bs_${bs}_opt_${opt}"
              bsub -n 4 -W 25:00 -R "rusage[mem=4096, ngpus_excl_p=2]" python train.py --lr $lr -b $bs -e 50 --op $opt --log $name
              sleep 1
        done
   done
done