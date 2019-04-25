import os
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random


class DataWrapper(Dataset):
    def __init__(self, input_dir, target_dir, cuda_available):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.cuda_available = cuda_available
        print(self.cuda_available)
        assert len([name for name in os.listdir(self.input_dir)]) == len([name for name in os.listdir(
            self.target_dir)]), "Input and target directory dont have the same number of entries"

    def __len__(self):
        return len([name for name in os.listdir(self.input_dir)])

    def __getitem__(self, idx):
        # TODO: Normalize to 0 -- 1
        def toTensorRGB(self, image):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            if self.cuda_available:
                torch_image = torch.from_numpy(image).type(torch.FloatTensor).cuda()
            else:
                torch_image = torch.from_numpy(image).type(torch.FloatTensor)
            return torch_image/255.

        def toTensorBW(self, image):
            torch_image = torch.from_numpy(image).view(400, 400, 1)
            torch_image = torch_image.permute((2, 0, 1))

            if self.cuda_available:
                torch_image = torch_image.type(torch.FloatTensor).cuda()
            else:
                torch_image = torch_image.type(torch.FloatTensor)
            return torch_image/255.

        input_img_name = os.path.join(self.input_dir, str(idx).zfill(5) + '.png')
        input_image = io.imread(input_img_name)
        target_img_name = os.path.join(self.target_dir, str(idx).zfill(5) + '.png')
        target_image = io.imread(target_img_name)
        sample = {'input': toTensorRGB(self, input_image), 'target': toTensorBW(self, target_image)}
        return sample


def create_batches(data, test_set, batch_size=10):
    if len(test_set) == 0:
        indices = range(len(data))
        training_indices = random.sample(indices, k=int(0.6 * len(data)))
        indices_2 = [x for x in indices if x not in training_indices]
        eval_indices = random.sample(indices_2, k=int(0.2 * len(data)))
        indices_3 = [x for x in indices_2 if x not in eval_indices]
        test_indices = indices_3
    else:
        indices = range(len(data))
        test_indices = test_set
        indices_2 = [x for x in indices if x not in test_indices]
        training_indices = random.sample(indices_2, k=int(0.6 * len(data)))
        indices_3 = [x for x in indices_2 if x not in training_indices]
        eval_indices = random.sample(indices_3, k=int(0.2 * len(data)))

    assert len(training_indices) + len(test_indices) + len(eval_indices) == len(data), "Not all data is used!"
    # create batches, shuffle needs to be false because we use the sampler.
    training_data = DataLoader(data, shuffle=False, batch_size=batch_size, sampler=SubsetRandomSampler(training_indices))
    val_data = DataLoader(data, shuffle=False, batch_size=239, sampler=SubsetRandomSampler(eval_indices))
    test_data = DataLoader(data, shuffle=False, batch_size=1, sampler=SubsetRandomSampler(test_indices))
    return [training_data, val_data, test_data, test_indices]

# input_dir = 'train_augmented/input/'
# target_dir = 'train_augmented/target/'
# data = DataWrapper(input_dir, target_dir)
# for i in range(len(data)):
#     sample = data[i]
#     print(i, sample['input'].shape, sample['target'].shape)
