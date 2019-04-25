#!/bin/bash

declare -a LearningRates=("1e-1", "1e-2", "1e-3", "1e-4", "1e-5")
declare -a BatchSizes=("2", "4", "8", "16", "32", "64")

# Iterate the string array using for loop
for lr in ${LearningRates[@]}; do
   for bs in ${BatchSizes[@]}; do
      name="lr_${lr}_bs_${bs}"
      bsub -n 4 -W 25:00 -R "rusage[mem=4096, ngpus_excl_p=2]" python train.py $lr $bs 50 $name
      sleep 1
   done
   sleep 3600
done