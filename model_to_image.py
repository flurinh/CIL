import torch
from DataWrapper import *
from PIL import Image
from architectures import *
import numpy as np
def save_image():

    return
input_dir = 'train_augmented/input/'
target_dir = 'train_augmented/target/'

data = DataWrapper(input_dir, target_dir, torch.cuda.is_available())
model = UNet(3,2)
model.load_state_dict(torch.load('models/test.pt'))
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model.cuda()
else:
    print("CUDA unavailable, using CPU!")
test_indices=[]
[training_data, val_data, test_data, test_indices] = create_batches(data, test_indices, batch_size=1)

for i_batch, batch in enumerate(training_data):
    print(i_batch)
    inputs = batch['input']
    outputs = model(inputs)
    outputs = outputs.cpu()
    outputs = outputs[0].view((400, 400)).detach().numpy()
    print(outputs)
    outputs = [[0. if pixel < 0.5 else 255. for pixel in row] for row in outputs]
    outputs = np.asarray(outputs)
    print(outputs)
    out_image = Image.fromarray(outputs)
    out_image.convert('RGB').save('output_images/'+str(i_batch).zfill(5)+".png")
