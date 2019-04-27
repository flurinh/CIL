#!/bin/bash

# Optimizers:      1: SGD
#                  2: Adam
#                  3: AdaDelta
#                  4: RMSProp
#
# TrainingSet:     1: Only train data
#                  2: Augmented train data
#                  3: Augmented + additional
#                  4: Augmented rescaled
#
# Model:           1: U_Net
#                  2: R2U_Net
#                  3: AttU_Net
#                  4: R2AttU_Net

declare -a LearningRates=("0.0001")
declare -a Optimizers=("2")
declare -a TrainingSet=("2")
declare -a BatchSizes=("1")
declare -a Models=("1")
declare -a Thresholds=("0.5")
# Iterate the string array using for loop
for lr in ${LearningRates[@]}; do
    for bs in ${BatchSizes[@]}; do
        for opt in ${Optimizers[@]}; do
            for ts in ${TrainingSet[@]}; do
                for thres in ${Thresholds[@]}; do
                    for model in ${Models[@]}; do
                        name="lr_${lr}_bs_${bs}_opt_${opt}_data_${ts}_model_${model}_thres_${thres}"
                        bsub -n 4 -W 25:00 -R "rusage[mem=4096, ngpus_excl_p=2]" python train.py \
                        --learning_rate $lr \
                        --batch_size $bs \
                        --nr_epochs 50 \
                        --optimizer $opt \
                        --data $ts \
                        --log_dir $name \
                        --model $model \
                        --thres $thres \
                        sleep 1
                    done
                done
            done
        done
    done
done