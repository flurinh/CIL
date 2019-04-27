from PIL import Image
import random
import glob
import os
import numpy as np
import shutil

def rotate_by_45(image_1):
    transformed_image = image_1
    # transformed_image = image_1.rotate(45)
    width, height = image_1.size
    left = (width - 282) / 2
    top = (height - 282) / 2
    right = (width + 282) / 2
    bottom = (height + 282) / 2

    transformed_image = transformed_image.crop((left, top, right, bottom)).resize((400, 400), resample=Image.BICUBIC)
    return transformed_image

def check_flawed(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    # Check for white pixels in the satellite image
    white = np.where(arr1 == 255)
    black = np.where(arr1 == 0)
    check1 = np.where(np.array(white[0]) == np.array(white[1]))[0].shape[0]
    check2 = np.where(np.array(black[0]) == np.array(black[1]))[0].shape[0]
    if (check1 > 100) or (check2 > 100):
        return False
    # Check for amount of street pixels in the map image
    elif np.where(arr2 == 1)[0].shape[0] < 100:
        return False
    else:
        return True

def fit_largest_rect(img1, dim):
    #Todo: return the best rotation, not the smallest (always possible solution)
    a = int(dim/4)
    b = int(dim/4)
    c = 3*a
    d = 3*b
    return (a, b, c, d) # (left, upper, right, lower)-tuple

def permute_img(img1, img2):
    dim = img1.size[0]
    angle = random.randint(0, 359)
    flips = [Image.FLIP_LEFT_RIGHT, 
             Image.FLIP_TOP_BOTTOM, 
             Image.TRANSPOSE]
    flip = flips[random.randint(0, 2)]
    img1 = img1.rotate(angle, expand=True).transpose(flip)
    img2 = img2.rotate(angle, expand=True).transpose(flip)
    rect = fit_largest_rect(img2, dim)
    img1 = img1.crop(rect).resize((dim, dim))
    img2 = img2.crop(rect).resize((dim, dim))
    return img1, img2

def crop_image(img1, img2, size, counter, val):
    num_w = img1.size[0] // size + 1
    num_h = img1.size[1] // size + 1
    new_w = num_w * size
    new_h = num_h * size
    img1 = img1.resize((new_w, new_h), Image.BICUBIC)
    img2 = img2.resize((new_w, new_h), Image.BICUBIC)
    idx = -1
    for i in range(num_w):
        for j in range(num_h):
            idx += 1
            box = (i*size, j*size, (i+1)*size, (j+1)*size)
            crop1 = img1.crop(box)
            crop2 = img2.crop(box).convert('1')
            if check_flawed(crop1, crop2) and not val:
                crop1.save(augmented_input_dir + str(counter).zfill(5) + '.png')
                crop2.save(augmented_target_dir + str(counter).zfill(5) + '.png')
                counter+=1
            elif check_flawed(crop1, crop2) and val:
                crop1.save(val_input_dir + str(counter).zfill(5) + '.png')
                crop2.save(val_target_dir + str(counter).zfill(5) + '.png')
                counter+=1
            if counter % 100 == 0:
                pass
                #print(counter)
    return counter

def transform_image_combined(image_1, image_2, counter, rescale=False, val=False, size = 400):
    '''
    :param image_1:
    :param image_2:
    :return:
    '''
    opened_image_1 = Image.open(image_1)
    opened_image_2 = Image.open(image_2)
    
    low_res = False
    if opened_image_1.size[0] < 1000:
        low_res = True

    if rescale is True and low_res is True:
        opened_image_1 = opened_image_1.resize((size, size), resample=Image.BICUBIC)
        opened_image_2 = opened_image_2.resize((size, size), resample=Image.BICUBIC)

    rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    flip_direction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE]
    if low_res:
        for angle in rotation_angles:
            for flip in flip_direction:
                transformed_image_input = opened_image_1.rotate(angle).transpose(flip)
                transformed_image_target = opened_image_2.rotate(angle).transpose(flip)
                transformed_image_target.convert('1')
                if angle % 90 != 0:
                    transformed_image_input = rotate_by_45(transformed_image_input)
                    transformed_image_target = rotate_by_45(transformed_image_target)
    
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
    else:
        counter = crop_image(opened_image_1, opened_image_2, size, counter, val)
    return counter

    # transformed_image_input = rotate_by_45(opened_image_1)
    # transformed_image_target = rotate_by_45(opened_image_2)
    # if val is False and rescale is False:
    #     transformed_image_input.save(augmented_input_dir + str(counter).zfill(5) + '.png')
    #     transformed_image_target.save(augmented_target_dir + str(counter).zfill(5) + '.png')
    #
    # if val is True and rescale is False:
    #     transformed_image_input.save(val_input_dir + str(counter).zfill(5) + '.png')
    #     transformed_image_target.save(val_input_dir + str(counter).zfill(5) + '.png')
    #
    # if val is False and rescale is True:
    #     transformed_image_input.save(rescaled_input_dir + str(counter).zfill(5) + '.png')
    #     transformed_image_target.save(rescaled_target_dir + str(counter).zfill(5) + '.png')
    #
    # if val is True and rescale is True:
    #     transformed_image_input.save(rescaled_val_input_dir + str(counter).zfill(5) + '.png')
    #     transformed_image_target.save(rescaled_val_input_dir + str(counter).zfill(5) + '.png')
    #
    # counter += 1
    return counter


# Decide what datasets to use: load_augmented_datasets = True will load the toronto dataset
load_augmented_datasets = True
# Define Validation split size:
validation_split = .8 # 80 % of the data is used for training
# point to the correct directories
original_root_dir = 'data/original/train'
original_input_dir = original_root_dir + '/input/'
original_target_dir = original_root_dir + '/target/'

toronto_dir = 'data/original/toronto'
toronto_input_dir = toronto_dir + '/input/'
toronto_target_dir = toronto_dir + '/target/'

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
if load_augmented_datasets:
    print("Loading all datasets (including the Toronto road segmentation dataset!)")
    original_input_images = glob.glob(original_input_dir + '*.png') + glob.glob(toronto_input_dir + '*.png')
    original_target_images = glob.glob(original_target_dir + '*.png') + glob.glob(toronto_target_dir + '*.png')
    print("Total images:", len(original_input_images))

else:
    original_input_images = glob.glob(original_input_dir + '*.png')
    original_target_images = glob.glob(original_target_dir + '*.png')

training_ind = random.sample(range(len(original_input_images)), k=int(validation_split * len(original_input_images)))
val_ind = [x for x in range(len(original_input_images)) if x not in training_ind]

std_counter = 0
scaled_counter = 0
validation_counter = 0
sc_val_counter = 0

print('Processing %d original training images.' % (len(training_ind)))
for i in training_ind:
    std_counter = transform_image_combined(original_input_images[i],
                                       original_target_images[i],
                                       std_counter)
    
    scaled_counter = transform_image_combined(original_input_images[i],
                                       original_target_images[i],
                                       scaled_counter,
                                       rescale=True,
                                       size = 608)

print('Processing %d original validation images.' % (len(val_ind)))
for j in val_ind:
    validation_counter = transform_image_combined(original_input_images[j],
                                       original_target_images[j],
                                       validation_counter,
                                       rescale=False,
                                       val=True)
    sc_val_counter = transform_image_combined(original_input_images[j],
                                       original_target_images[j],
                                       sc_val_counter,
                                       rescale=True,
                                       val=True,
                                       size = 608)

print("Standard-process images:",std_counter)
print("Scaled images:", scaled_counter)
print("Std validation images:", validation_counter)
print("Scaled validation images:", sc_val_counter)