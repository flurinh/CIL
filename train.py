import torch
import tensorboardX
from DataWrapper import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import numpy as np
from architectures import *
import torch.nn as nn
import torch.optim
from plotter_helper import *

seed = 42
# np.random.seed(seed)
torch.manual_seed(seed)

input_dir = 'train_augmented/input/'
target_dir = 'train_augmented/target/'

data = DataWrapper(input_dir, target_dir, torch.cuda.is_available())

model = SimpleCNN()
if torch.cuda.is_available():
    model.cuda()
else:
    print("CUDA unavailable, using CPU!")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
number_of_epochs = 10
test_indices = []
for n in range(number_of_epochs):
    [training_data, val_data, test_data, test_indices] = create_batches(data, test_indices)
    print("Starting Epoch:\t", n)
    for i_batch, batch in enumerate(training_data):
        inputs = batch['input']
        outputs = model(inputs)
        loss = criterion(batch['target'], outputs)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        print("Epoch:\t", n, "\t Batch:\t", i_batch, "\tof\t", len(training_data))

print("Done Training -- Starting Evaluation")
for i_batch, batch in enumerate(test_data):
    if i_batch > 1:
        break
    inputs = batch['input']
    outputs = model(inputs)
    groundtruth = batch['target']
    outputs = outputs.cpu()
    outputs = outputs[0].view((400, 400)).detach().numpy()
    outputs = [[0 if pixel < 0.5 else 1 for pixel in row] for row in outputs]
    evaluation_side_by_side_plot(inputs.cpu(), outputs, groundtruth.cpu())
