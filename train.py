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

LEARNING_RATE = 1e-3
BATCH_SIZE = 3

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

input_dir = 'train_augmented/input/'
target_dir = 'train_augmented/target/'

data = DataWrapper(input_dir, target_dir, torch.cuda.is_available())

model = UNet(3, 2)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model.cuda()
else:
    print("CUDA unavailable, using CPU!")

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),
                                  lr=LEARNING_RATE,
                                  momentum=0.9,
                                  weight_decay=0.0005)
number_of_epochs = 3
test_indices = []
mean_losses = []
figure = plt.figure()
for n in range(number_of_epochs):
    [training_data, val_data, test_data, test_indices] = create_batches(data, test_indices, batch_size=BATCH_SIZE)
    print("Starting Epoch:\t", n)
    losses = []
    for i_batch, batch in enumerate(training_data):
        optimizer.zero_grad()
        inputs = batch['input']
        outputs = model(inputs)
        loss = criterion(outputs, batch['target'])
        loss.backward()
        optimizer.step()
        print("Epoch:\t", n, "\t Batch:\t", i_batch, "\tof\t", len(training_data))
        torch.save(model.state_dict(), 'models/test.pt')
        losses.append(loss.cpu().detach().numpy())

    mean_losses.append(np.mean(losses))
    plt.clf()
    plt.plot(mean_losses)
    plt.show()
    plt.pause(0.01)

print("Done Training -- Starting Evaluation")
for i_batch, batch in enumerate(test_data):
    if i_batch > 1:
        break
    inputs = batch['input']
    outputs = model(inputs)
    groundtruth = batch['target']
    outputs = outputs.cpu()
    outputs = outputs[0].view((400, 400)).detach().numpy()
    print(outputs)
    outputs = [[0. if pixel < 0.5 else 1. for pixel in row] for row in outputs]
    print(outputs)
    print(groundtruth.cpu())
    evaluation_side_by_side_plot(inputs.cpu(), outputs, groundtruth.cpu())
