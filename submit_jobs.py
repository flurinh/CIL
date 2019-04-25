import os
import time

learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
batch_sizes = [2, 4, 8, 16, 32, 64]
n_epochs = 50

for lr in learning_rates:
    for bs in batch_sizes:
        name = "lr_" + str(lr) + "_bs_" + str(bs)
        os.system('bsub -n 4 -W 25:00 -R "rusage[mem=4096, ngpus_excl_p=2]" python train.py ' + str(lr) + ' ' + str(bs) + ' ' + str(n_epochs) + ' ' + name)
        time.sleep(1)
    time.sleep(3600)
