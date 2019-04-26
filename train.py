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
import json
import argparse
import time

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--lr", nargs="?", type=float, dest="learning_rate", default="0.0005",
                    help="Learning rate of the model as float")
parser.add_argument("--op", nargs="?", type=int, dest="optimizer", default="2",
                    help="Optimizer to use: \n"
                         "1: SGD\n"
                         "2: Adam\n"
                         "3: AdaDelta\n"
                         "4: RMSProp")
parser.add_argument("-d", nargs="?", type=int, dest="dataset", default="2",
                    help="Dataset to use: \n"
                         "1: Only our training data\n"
                         "2: Augmented training data\n"
                         "3: Augmented training data + additional data (Thomas)")
parser.add_argument("-b", nargs="?", type=int, dest="batch_size", default="1",
                    help="Batch size")
parser.add_argument("--log", nargs="?", type=str, dest="log_dir", default="model",
                    help="Log directory")
parser.add_argument("-e", nargs="?", type=int, dest="nr_epochs", default="50",
                    help="Number of epochs")

args = parser.parse_args()

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
NUMBER_EPOCHS = args.nr_epochs
OPTIMIZER = args.optimizer
TRAIN_SET = args.dataset
LOG_NAME = args.log_dir + "_" + str(int(time.time()))

writer = SummaryWriter('logdir/' + LOG_NAME)
json_saver = {'train_loss': dict(), 'val_loss': dict(), 'n_parameters': 0, 'test_indices': []}

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

if TRAIN_SET is 1:
    input_dir = 'training/input/'
    target_dir = 'training/target/'
    val_input_dir = 'val/input/'
    val_target_dir = 'val/target/'

elif TRAIN_SET is 2:
    input_dir = 'train_augmented/input/'
    target_dir = 'train_augmented/target/'
    val_input_dir = 'val/input/'
    val_target_dir = 'val/target/'

elif TRAIN_SET is 3:
    input_dir = 'DeepGlobe/input/'
    target_dir = 'DeepGlobe/target/'

elif TRAIN_SET is 4:
    input_dir = 'train_rescaled/input/'
    target_dir = 'train_rescaled/target/'
    val_input_dir = 'val_rescaled/input/'
    val_target_dir = 'val_rescaled/target/'


train_data = DataWrapper(input_dir, target_dir, torch.cuda.is_available())
val_data = DataWrapper(val_input_dir, val_target_dir, torch.cuda.is_available())

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
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.0005)

elif OPTIMIZER is 4:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-08, weight_decay=0.0005,
                                    momentum=0.9,
                                    centered=False)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("number of trainable paramters in model:", params)
writer.add_text("Trainable Parameters", str(params))
json_saver['n_parameters'] = int(params)
test_indices = []
mean_losses = []
figure = plt.figure()
best_val = np.inf

for n in range(NUMBER_EPOCHS):
    training_data = create_batches(train_data, batch_size=BATCH_SIZE)
    test_data = create_batches(val_data, batch_size=1)
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
    json_saver['train_loss'][str(n)] = float(np.mean(losses))

    with torch.no_grad():
        val_loss = 0
        for i_batch, batch in enumerate(test_data):
            model.eval()
            inputs = batch['input']
            print(inputs.size())
            outputs = model(inputs)
            if TRAIN_SET is 4:
                outputs = outputs[0].cpu().view((608, 608)).detach().numpy()
                ground = batch['target'].cpu().view((608, 608)).detach().numpy()
            else:
                outputs = outputs[0].cpu().view((400, 400)).detach().numpy()
                ground = batch['target'].cpu().view((400, 400)).detach().numpy()
            outputs = np.asarray([[0. if pixel < 0.5 else 1. for pixel in row] for row in outputs])
            diff = outputs - ground
            squared = np.square(diff)
            accuracy = np.sum(squared) / diff.size
            val_loss += accuracy

    val_loss /= len(test_data)
    writer.add_scalar('Validation Loss', float(val_loss), n)
    json_saver['val_loss'][str(n)] = float(val_loss)

    if val_loss < best_val:
        torch.save(model.state_dict(), 'models/' + LOG_NAME + '.pt')
        best_val = val_loss

    with open('logdir/' + LOG_NAME + '.json', 'w') as fp:
        json.dump(json_saver, fp)

# print("Done Training -- Starting Evaluation")
# for i_batch, batch in enumerate(test_data):
#     if i_batch > 1:
#         break
#     inputs = batch['input']
#     outputs = model(inputs)
#     target = batch['target']
#     outputs = outputs.cpu()
#     outputs = outputs[0].view((400, 400)).detach().numpy()
#     print(outputs)
#     outputs = [[0. if pixel < 0.5 else 1. for pixel in row] for row in outputs]
#     print(outputs)
#     print(target.cpu())
#     evaluation_side_by_side_plot(inputs.cpu(), outputs, target.cpu())
