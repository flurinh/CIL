import os
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt
import torch


class DataWrapper(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        assert len([name for name in os.listdir(self.input_dir)]) == len([name for name in os.listdir(
            self.target_dir)]), "Input and target directory dont have the same number of entries"

    def __len__(self):
        return len([name for name in os.listdir(self.input_dir)])

    def __getitem__(self, idx):
        def toTensorRGB(image):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            return torch.from_numpy(image)

        input_img_name = os.path.join(self.input_dir, str(idx).zfill(5) + '.png')
        input_image = io.imread(input_img_name)
        target_img_name = os.path.join(self.target_dir, str(idx).zfill(5) + '.png')
        target_image = io.imread(target_img_name)
        sample = {'input': toTensorRGB(input_image), 'target': torch.from_numpy(target_image)}
        return sample

# input_dir = 'train_augmented/input/'
# target_dir = 'train_augmented/target/'
#
# data=Data(input_dir, target_dir)
#
# print(len(data))
# print(data[0]['target'])
# plt.imshow(data[0]['target'])
# plt.show()
# for i in range(len(data)):
#     sample = data[i]
#     print(i, sample['input'].shape, sample['target'].shape)
