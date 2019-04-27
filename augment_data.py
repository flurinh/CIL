from PIL import Image
import random
import glob
import os
import numpy as np
import shutil

def rotate_by_45(image_1):
    transformed_image = image_1.rotate(45)
    width, height = image_1.size
    left = (width - 282) / 2
    top = (height - 282) / 2
    right = (width + 282) / 2
    bottom = (height + 282) / 2

    transformed_image = transformed_image.crop((left, top, right, bottom)).resize((400, 400),
                                                                                                resample=Image.BICUBIC)
    return transformed_image


def transform_image_combined(image_1, image_2, counter, rescale=False, val=False):
    '''
    :param image_1:
    :param image_2:
    :return:
    '''

    opened_image_1 = Image.open(image_1)
    opened_image_2 = Image.open(image_2)

    if rescale is True:
        opened_image_1 = opened_image_1.resize((608, 608), resample=Image.BICUBIC)
        opened_image_2 = opened_image_2.resize((608, 608), resample=Image.BICUBIC)

    rotation_angles = [0, 90, 180, 270]
    flip_direction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE]
    for angle in rotation_angles:
        for flip in flip_direction:
            transformed_image_input = opened_image_1.rotate(angle).transpose(flip)
            transformed_image_target = opened_image_2.rotate(angle).transpose(flip)
            transformed_image_target.convert('1')
            if val is False and rescale is False:
                transformed_image_input.save(augmented_input_dir + str(counter).zfill(5) + '.png')
                transformed_image_target.save(augmented_target_dir + str(counter).zfill(5) + '.png')
            if val is True and rescale is False:
                transformed_image_input.save(val_input_dir + str(counter).zfill(5) + '.png')
                transformed_image_target.save(val_target_dir + str(counter).zfill(5) + '.png')

            if val is False and rescale is True:
                transformed_image_input.save(rescaled_input_dir + str(counter).zfill(5) + '.png')
                transformed_image_target.save(rescaled_target_dir + str(counter).zfill(5) + '.png')

            if val is True and rescale is True:
                transformed_image_input.save(rescaled_val_input_dir + str(counter).zfill(5) + '.png')
                transformed_image_target.save(rescaled_val_target_dir + str(counter).zfill(5) + '.png')
            counter += 1
            if counter % 100 == 0:
                print(counter)
    transformed_image_input = rotate_by_45(opened_image_1)
    transformed_image_target = rotate_by_45(opened_image_2)
    if val is False and rescale is False:
        transformed_image_input.save(augmented_input_dir + str(counter).zfill(5) + '.png')
        transformed_image_target.save(augmented_target_dir + str(counter).zfill(5) + '.png')
        counter += 1
    if val is True and rescale is False:
        transformed_image_input.save(val_input_dir + str(counter).zfill(5) + '.png')
        transformed_image_target.save(val_input_dir + str(counter).zfill(5) + '.png')
        counter += 1
    if val is False and rescale is True:
        transformed_image_input.save(rescaled_input_dir + str(counter).zfill(5) + '.png')
        transformed_image_target.save(rescaled_target_dir + str(counter).zfill(5) + '.png')
        counter += 1
    if val is True and rescale is True:
        transformed_image_input.save(rescaled_val_input_dir + str(counter).zfill(5) + '.png')
        transformed_image_target.save(rescaled_val_input_dir + str(counter).zfill(5) + '.png')
        counter += 1

    return counter


# point to the correct directories
original_root_dir = 'data/original/train'
original_input_dir = original_root_dir + '/input/'
original_target_dir = original_root_dir + '/target/'

augmented_root_dir = 'data/augmented/train'
augmented_input_dir = augmented_root_dir + '/input/'
augmented_target_dir = augmented_root_dir + '/target/'

val_root_dir = 'data/augmented/validate'
val_input_dir = val_root_dir + '/input/'
val_target_dir = val_root_dir + '/target/'

rescaled_root_dir = 'data/scaled/train'
rescaled_input_dir = rescaled_root_dir + '/input/'
rescaled_target_dir = rescaled_root_dir + '/target/'

rescaled_val_root_dir = 'data/scaled/validate'
rescaled_val_input_dir = rescaled_val_root_dir + '/input/'
rescaled_val_target_dir = rescaled_val_root_dir + '/target/'

for name in ['data/augmented', 'data/scaled', augmented_root_dir, augmented_input_dir, augmented_target_dir, val_root_dir, val_input_dir,
             val_target_dir, rescaled_root_dir, rescaled_val_root_dir, rescaled_input_dir, rescaled_target_dir,
             rescaled_val_input_dir, rescaled_val_target_dir]:

    if os.path.isdir(name):
        shutil.rmtree(name)
    os.mkdir(name)

# load input
original_input_images = glob.glob(original_input_dir + '*.png')
original_target_images = glob.glob(original_target_dir + '*.png')

training_ind = random.sample(range(len(original_input_images)), k=int(0.8 * len(original_input_images)))
val_ind = [x for x in range(len(original_input_images)) if x not in training_ind]
counter = 0

for i in training_ind:
    counter = transform_image_combined(original_input_images[i],
                                       original_target_images[i],
                                       counter)

counter = 0
for i in training_ind:
    counter = transform_image_combined(original_input_images[i],
                                       original_target_images[i],
                                       counter,
                                       rescale=True)

counter = 0
for i in val_ind:
    counter = transform_image_combined(original_input_images[i],
                                       original_target_images[i],
                                       counter,

                                       val=True)

counter = 0
for i in val_ind:
    counter = transform_image_combined(original_input_images[i],
                                       original_target_images[i],
                                       counter,
                                       rescale=True,
                                       val=True)
