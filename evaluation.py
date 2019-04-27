import os
from PIL import Image
import torch
import numpy as np
from skimage import io
from plotter_helper import *
from mask_to_submission import *

def evaluate(save_dir, model, threshold=0.5):
    def toTensorRGB(image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_1 = image.transpose((2, 0, 1))
        torch_image = torch.from_numpy(np.asarray([image_1])).type(torch.FloatTensor)
        return torch_image / 255.

    prediction_test_dir = save_dir + "/results/prediction"
    for i in range(1, 224):
        filename = "data/original/test/test_" + str(i) + ".png"
        if not os.path.isfile(filename):
            continue
        print("Loading image {}".format(filename))

        # Only prediction
        img = io.imread(filename)
        input = toTensorRGB(img)
        outputs = model(input)
        outputs = outputs[0].view((img.shape[0], img.shape[1])).detach().numpy()
        outputs = [[0. if pixel < threshold else 255. for pixel in row] for row in outputs]
        outputs = np.asarray(outputs)
        out_image = Image.fromarray(outputs)
        out_image.convert('RGB').save(prediction_test_dir + str(i) + ".png")
    return


def create_overlays(save_dir):
    for i in range(1, 224):
        filename = save_dir+"/results/prediction/" + str(i) + ".png"
        filename_im = "data/original/test/test_" + str(i) + ".png"
        if not os.path.isfile(filename):
            continue

        # Only prediction
        prediction = Image.open(filename)
        prediction = prediction.convert('L')
        prediction = np.asarray(prediction)
        image = io.imread(filename_im)
        overlay_image = make_img_overlay(image, prediction)
        overlay_image.save(save_dir+"results/overlay/" + str(i) + ".png")
    return


def mask2submission(submission_filename, prediction_directory):
    if not os.path.isdir(prediction_directory):
        print("No directory found. Run the predictions first")

    image_filenames = []
    for i in range(1, 244):
        filename = prediction_directory + str(i) + ".png"
        if not os.path.isfile(filename):
            continue
        image_filenames.append(filename)
    masks_to_submission(submission_filename, *image_filenames)
    return
