import torch
from DataWrapper import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import numpy as np
from architectures import *
import torch.nn as nn
import torch.optim
from plotter_helper import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('logdir/exp-1')

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUMBER_EPOCHS = 1000

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

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("number of trainable paramters in model:", params)

test_indices = []
mean_losses = []
figure = plt.figure()
best_val = np.inf

dummy_input = (torch.zeros(3, 400, 400),)

for n in range(NUMBER_EPOCHS):
    [training_data, val_data, test_data, test_indices] = create_batches(data, test_indices, batch_size=BATCH_SIZE)
    for i, entry in enumerate(val_data):
        print(i)

    print("Starting Epoch:\t", n)
    losses = []
    model.train()
    for i_batch, batch in enumerate(training_data):
        optimizer.zero_grad()
        inputs = batch['input']
        outputs = model(inputs)
        loss = criterion(outputs, batch['target'])
        loss.backward()
        optimizer.step()
        print("Epoch:\t", n, "\t Batch:\t", i_batch, "\tof\t", len(training_data))
        losses.append(loss.cpu().detach().numpy())
    losses=[]
    writer.add_scalar('Training Loss', float(np.mean(mean_losses)), n)
    val_loss = 0
    for i_batch, batch in enumerate(val_data):
        model.eval()
        inputs = batch['input']
        print(inputs.size())
        outputs = model(inputs)
        loss = criterion(outputs, batch['target'])
        val_loss += loss

    val_loss /= len(val_data)
    writer.add_scalar('Validation Loss', val_loss, n)

    if val_loss < best_val:
        writer.add_graph("Best Model", model, dummy_input)
        torch.save(model, 'models/best.pt')
        best_val = val_loss

# print("Done Training -- Starting Evaluation")
# for i_batch, batch in enumerate(test_data):
#     if i_batch > 1:
#         break
#     inputs = batch['input']
#     outputs = model(inputs)
#     groundtruth = batch['target']
#     outputs = outputs.cpu()
#     outputs = outputs[0].view((400, 400)).detach().numpy()
#     print(outputs)
#     outputs = [[0. if pixel < 0.5 else 1. for pixel in row] for row in outputs]
#     print(outputs)
#     print(groundtruth.cpu())
#     evaluation_side_by_side_plot(inputs.cpu(), outputs, groundtruth.cpu())
