#!/bin/bash

declare -a LearningRates=("0.1", "0.01", "0.001", "0.0001", "0.00001")
declare -a BatchSizes=("2", "4", "8", "16", "32", "64")

# Iterate the string array using for loop
for lr in ${LearningRates[@]}; do
   for bs in ${BatchSizes[@]}; do
      name="lr_${lr}_bs_${bs}"
      bsub -n 4 -W 25:00 -R "rusage[mem=4096, ngpus_excl_p=2]" python train.py $lr $bs 80 $name
      sleep 1
   done
   sleep 3600
done