import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as mpimg


def evaluation_side_by_side_plot(inputs, outputs, groundtruth, save=False, save_name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(inputs[0].permute((1, 2, 0)).numpy())
    ax1.set_title("Input")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(outputs, cmap='Greys_r')
    ax2.set_title("Output")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(groundtruth[0].view((400, 400)), cmap='Greys_r')
    ax3.set_title("Groundtruth")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)


def evaluation_side_by_side_plot_np(inputs, outputs, groundtruth, save=False, save_name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(inputs)
    ax1.set_title("Input")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(outputs, cmap='Greys_r')
    ax2.set_title("Output")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(groundtruth, cmap='Greys_r')
    ax3.set_title("Groundtruth")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = 255 * predicted_img
    color_mask[:, :, 1] = 165 * predicted_img

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    img9 = img_float_to_uint8(predicted_img)
    alpha = Image.fromarray(255 - img9, "L")

    blended_overlay = Image.blend(background, overlay, 0.7)
    new_img = Image.composite(background, blended_overlay, alpha)
    return new_img


def overlay_side_by_side(img, ground_truth, prediction, save=False, save_name="plot.png"):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(make_img_overlay(img, ground_truth))
    ax1.set_title("Ground truth")
    ax2.imshow(make_img_overlay(img, prediction))
    ax2.set_title("Prediction")
    if not save:
        plt.show()
    else:
        fig.savefig(save_name)
    plt.close(fig)

if __name__ == "__main__":
    image = mpimg.imread("training/images/satImage_001.png")
    ground_truth = mpimg.imread("training/target/satImage_001.png")
    overlay_image = overlay_side_by_side(image, ground_truth, ground_truth)