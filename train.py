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
from architecture_2 import *

# TODO: MAKE SELECTION VIA BASH ON MODEL
# TODO: SAVE MODEL NAME TO JSON

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--learning_rate", nargs="?", type=float, dest="learning_rate", default="0.0005",
                    help="Learning rate of the model as float")
parser.add_argument("--optimizer", nargs="?", type=int, dest="optimizer", default="2",
                    help="Optimizer to use: \n"
                         "1: SGD\n"
                         "2: Adam\n"
                         "3: AdaDelta\n"
                         "4: RMSProp")
parser.add_argument("--data", nargs="?", type=int, dest="dataset", default="2",
                    help="Dataset to use: \n"
                         "1: Only our train data\n"
                         "2: Augmented train data\n"
                         "3: Augmented train data + additional data (Thomas)\n"
                         "4: Augmented train data rescaled to (608,608)")
parser.add_argument("--batch_size", nargs="?", type=int, dest="batch_size", default="1",
                    help="Batch size")
parser.add_argument("--log_dir", nargs="?", type=str, dest="log_dir", default="model",
                    help="Log directory")
parser.add_argument("--nr_episodes", nargs="?", type=int, dest="nr_epochs", default="50",
                    help="Number of epochs")
parser.add_argument("--model", nargs="?", type=int, dest="model", default="1",
                    help="Model to run:\n"
                         "1: U_Net\n"
                         "2: R2U_Net\n"
                         "3: AttU_Net\n"
                         "4: R2AttU_Net")
parser.add_argument("--thres", nargs="?", type=float, dest="threshold", default="0.5",
                    help="Threshold for the validation set and test set")

args = parser.parse_args()

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
NUMBER_EPOCHS = args.nr_epochs
OPTIMIZER = args.optimizer
TRAIN_SET = args.dataset
LOG_NAME = args.log_dir + "_" + str(int(time.time()))
MODEL = args.model
THRESHOLD = args.threshold

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

print("CREATING FOLDER STRUCTURE")
data_dir = 'data/'
if os.path.isdir('runs') is False:
    os.mkdir('runs')

if os.path.isdir('runs/' + LOG_NAME):
    save_dir = 'runs/' + LOG_NAME + str(int(time.time()))

else:
    save_dir = 'runs/' + LOG_NAME
model_dir = save_dir + '/model'
results_dir = save_dir + '/results'
os.mkdir(save_dir)
os.mkdir(model_dir)
os.mkdir(results_dir)

print("CREATING LOG FILES")
writer = SummaryWriter('logdir/' + LOG_NAME)
json_saver = {'train_loss': dict(),
              'val_loss': dict(),
              'n_parameters': 0,
              'start_time': int(time.time()),
              'end_time': 0,
              'run_time': 0,
              'name': LOG_NAME,
              'dataset': TRAIN_SET,
              'number_epochs': NUMBER_EPOCHS,
              'optimizer': OPTIMIZER,
              'learningrate': LEARNING_RATE,
              'train_set': TRAIN_SET,
              'batchsize': BATCH_SIZE,
              'model_save_epoch': 0}
with open(save_dir + 'data.json', 'w') as fp:
    json.dump(json_saver, fp)

if TRAIN_SET is 1:
    root_dir = 'original'
    input_dir = root_dir + '/train/input/'
    target_dir = root_dir + '/train/target/'
    val_input_dir = root_dir + '/validate/input/'
    val_target_dir = root_dir + '/validate/target/'

elif TRAIN_SET is 2:
    root_dir = 'augmented'
    input_dir = root_dir + '/train/input/'
    target_dir = root_dir + '/train/target/'
    val_input_dir = root_dir + '/validate/input/'
    val_target_dir = root_dir + '/validate/target/'

elif TRAIN_SET is 3:
    root_dir = 'DeepGlobe'
    input_dir = root_dir + '/train/input/'
    target_dir = root_dir + '/train/target/'
    input_dir = root_dir + '/validate/input/'
    target_dir = root_dir + '/validate/target/'

elif TRAIN_SET is 4:
    root_dir = 'scaled'
    input_dir = root_dir + '/train/input/'
    target_dir = root_dir + '/train/target/'
    val_input_dir = root_dir + '/validate/input/'
    val_target_dir = root_dir + '/validate/target/'

train_data = DataWrapper(data_dir + input_dir, data_dir + target_dir, torch.cuda.is_available())
val_data = DataWrapper(data_dir + val_input_dir, data_dir + val_target_dir, torch.cuda.is_available())

if MODEL is 1:
    model = UNet(3, 2)
elif MODEL is 2:
    model = R2U_Net()
elif MODEL is 3:
    model = AttU_Net()
elif MODEL is 4:
    model = R2AttU_Net()
else:
    print("Not a valid model")
    exit(1)

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
    validation_data = create_batches(val_data, batch_size=1)
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
            print("Epoch:\t", n, "\tof\t", NUMBER_EPOCHS, "\t Batch:\t", i_batch, "\tof\t", len(training_data))
        losses.append(loss.cpu().detach().numpy())

    writer.add_scalar('Training Loss', float(np.mean(losses)), n)
    json_saver['train_loss'][str(n)] = float(np.mean(losses))

    with torch.no_grad():
        val_loss = 0
        for i_batch, batch in enumerate(validation_data):
            model.eval()
            inputs = batch['input']
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

    val_loss /= len(validation_data)
    writer.add_scalar('Validation Loss', float(val_loss), n)
    json_saver['val_loss'][str(n)] = float(val_loss)

    if val_loss < best_val:
        torch.save(model.state_dict(), model_dir + '/model.pt')
        json_saver['model_save_epoch'] = n
        best_val = val_loss

    with open(save_dir + '/data.json', 'w') as fp:
        json.dump(json_saver, fp)

json_saver['end_time'] = int(time.time())
json_saver['run_time'] = json_saver['end_time'] - json_saver['start_time']
print("DONE TRAINING IN\t" + str(json_saver['run_time']) + "/t SECONDS")

with open(save_dir + '/data.json', 'w') as fp:
    json.dump(json_saver, fp)
print("SAVED TRAINING DATA")
print("STARTING EVALUATION")


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
