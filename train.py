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
import sys
from skimage import io


LEARNING_RATE = float(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
NUMBER_EPOCHS = int(sys.argv[3])
OPTIMIZER = int(sys.argv[4])
LOG_NAME = str(sys.argv[5])

writer = SummaryWriter('logdir/' + LOG_NAME)

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
if OPTIMIZER is 1:
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=0.9,
                                weight_decay=0.0005)

elif OPTIMIZER is 2:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

elif OPTIMIZER is 3:
    torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.0005)

elif OPTIMIZER is 4:
    torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08, weight_decay=0.0005, momentum=0.9, centered=False)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("number of trainable paramters in model:", params)
writer.add_text("Trainable Parameters", str(params))

test_indices = []
mean_losses = []
figure = plt.figure()
best_val = np.inf

dummy_input = torch.zeros(1, 3, 400, 400)

for n in range(NUMBER_EPOCHS):
    [training_data, val_data, test_data, test_indices] = create_batches(data, test_indices, batch_size=BATCH_SIZE)
    print("Starting Epoch:\t", n)
    losses = []
    model.train()
    for i_batch, batch in enumerate(training_data):
        optimizer.zero_grad()
        model.eval()
        inputs = batch['input']
        outputs = model(inputs)
        loss = criterion(outputs, batch['target'])
        loss.backward()
        optimizer.step()
        if i_batch % 100 == 0:
            print("Epoch:\t", n, "\t Batch:\t", i_batch, "\tof\t", len(training_data))
        losses.append(loss.cpu().detach().numpy())

    writer.add_scalar('Training Loss', float(np.mean(losses)), n)
    with torch.no_grad():
        val_loss = 0
        for i_batch, batch in enumerate(val_data):
            model.eval()
            inputs = batch['input']
            print(inputs.size())
            outputs = model(inputs)
            outputs = outputs[0].cpu().view((400, 400)).detach().numpy()
            outputs = np.asarray([[0. if pixel < 0.5 else 1. for pixel in row] for row in outputs])
            diff = outputs - batch['target'].cpu().view((400, 400)).detach().numpy()
            squared = np.square(diff)
            accuracy = np.sum(squared)/diff.size
            val_loss += accuracy

    val_loss /= len(val_data)
    writer.add_scalar('Validation Loss', float(val_loss), n)

    if val_loss < best_val:
        writer.add_graph(LOG_NAME, model, dummy_input)
        torch.save(model, 'models/' + LOG_NAME + '.pt')
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
