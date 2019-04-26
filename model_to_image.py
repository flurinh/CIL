import torch
from DataWrapper import *
from PIL import Image
from architectures import *
import numpy as np
import matplotlib.image as mpimg
from torch.utils.data import DataLoader

def toTensorRGB(image):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image_1 = image.transpose((2, 0, 1))
    torch_image = torch.from_numpy(np.asarray([image_1])).type(torch.FloatTensor)
    return torch_image / 255.


input_dir = 'test/'
target_dir = 'train_augmented/target/'

model = UNet(3, 2)
model.load_state_dict(torch.load('models/model_1556280564.pt'))
model.eval()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     model.cuda()
# else:
#     print("CUDA unavailable, using CPU!")

prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)



for i in range(1, 224):
    filename = "test/test_" + str(i) + ".png"
    if not os.path.isfile(filename):
        continue
    print("Loading image {}".format(filename))

    # Only prediction
    img = io.imread(filename)
    input = toTensorRGB(img)
    outputs = model(input)

    # outputs = outputs.cpu()
    outputs = outputs[0].view((608, 608)).detach().numpy()
    outputs = [[0. if pixel < 0.5 else 255. for pixel in row] for row in outputs]
    outputs = np.asarray(outputs)
    out_image = Image.fromarray(outputs)
    out_image.convert('RGB').save(prediction_test_dir + 'test_prediction_' + str(i) + ".png")
