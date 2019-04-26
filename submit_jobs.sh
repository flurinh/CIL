#!/bin/bash

declare -a LearningRates=("0.0005")
declare -a BatchSizes=("1" "2")
declare -a Optimizers=("1" "2" "3" "4")
# Iterate the string array using for loop
for lr in ${LearningRates[@]}; do
   for bs in ${BatchSizes[@]}; do
        for opt in ${Optimizers[@]}; do
              name="lr_${lr}_bs_${bs}_opt_${opt}"
              bsub -n 4 -W 25:00 -R "rusage[mem=4096, ngpus_excl_p=3]" python train.py --lr $lr -b $bs -e 50 --op $opt --log $name
              sleep 1
        done
   done
done