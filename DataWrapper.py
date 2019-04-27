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
            torch_image = torch.from_numpy(image).view(image.shape[0], image.shape[1], 1)
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


def create_batches(data, batch_size=10):
    # create batches, shuffle needs to be false because we use the sampler.
    data = DataLoader(data, shuffle=True, batch_size=batch_size)
    return data

# input_dir = 'train_augmented/input/'
# target_dir = 'train_augmented/target/'
# data = DataWrapper(input_dir, target_dir)
# for i in range(len(data)):
#     sample = data[i]
#     print(i, sample['input'].shape, sample['target'].shape)
