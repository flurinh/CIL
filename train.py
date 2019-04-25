import torch
import tensorboardX
from DataWrapper import *
from torch.utils.data import  DataLoader, SubsetRandomSampler
import random
import numpy as np
from models import  *
import torch.nn as nn
import torch.optim
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)



input_dir = 'train_augmented/input/'
target_dir = 'train_augmented/target/'

data = DataWrapper(input_dir, target_dir)

# generate indices for creating train, eval and test set.
indices = range(len(data))
training_indices = random.sample(indices, k=int(0.6 * len(data)))
indices_2 = [x for x in indices if x not in training_indices]
eval_indices = random.sample(indices_2, k=int(0.2 * len(data)))
indices_3 = [x for x in indices_2 if x not in eval_indices]
test_indices = indices_3
assert len(training_indices) + len(test_indices) + len(eval_indices) == len(data), "Not all data is used!"

# create batches, shuffle needs to be false because we use the sampler.
training_data = DataLoader(data, shuffle=False, batch_size=10, num_workers=4, sampler=SubsetRandomSampler(training_indices))
eval_data = DataLoader(data, shuffle=False, batch_size=1, num_workers=4, sampler=SubsetRandomSampler(eval_indices))
test_data = DataLoader(data, shuffle=False, batch_size=1, num_workers=4, sampler=SubsetRandomSampler(test_indices))

model = SimpleCNN()
# if print(torch.cuda.is_available()) is False:
#     print("CUDA unavailable, using CPU!")
# else:
#     model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

for i_batch, batch in enumerate(training_data):
    print(i_batch, batch['input'].size(), batch['target'].size())
    # get the inputs
    inputs = batch['input']
    # zero the parameter gradients
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.size())