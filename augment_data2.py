from PIL import Image
import PIL
import random
import glob
import os
import numpy as np

def check_flawed(img1, img2):
    # Todo: implement function that returns True if the picture is not flawed
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    # Check for completely white pixels in the satellite image
    white = np.where(arr1 == 255)
    check1 = np.where(np.array(white[0]) == np.array(white[1]))[0].shape[0]
    if check1 > 100:
        return False
    # Check for amount of street pixels in the map image
    elif np.where(arr2 == 1)[0].shape[0] < 100:
        return False
    else:
        return True

def crop_image(img1, img2, size, counter):
    num_w = img1.size[0] // size + 1
    num_h = img1.size[1] // size + 1
    new_w = num_w * size
    new_h = num_h * size
    img1 = img1.resize((new_w, new_h), PIL.Image.ANTIALIAS)
    img2 = img2.resize((new_w, new_h), PIL.Image.ANTIALIAS)
    idx = -1
    for i in range(num_w):
        for j in range(num_h):
            idx += 1
            box = (i*size, j*size, (i+1)*size, (j+1)*size)
            crop1 = img1.crop(box)
            crop2 = img2.crop(box).convert('1')
            if check_flawed(crop1, crop2):
                crop1.save(augmented_input_dir + str(counter).zfill(5) + '.png')
                crop2.save(augmented_target_dir + str(counter).zfill(5) + '.png')
                counter+=1
                if counter % 100 == 0:
                    print(counter)
    return counter

def transform_image_combined(image_1, image_2, counter, size=400):
    '''
    :param image_1:
    :param image_2:
    :return:
    '''
    opened_image_1 = Image.open(image_1)
    opened_image_2 = Image.open(image_2)
    rotation_angles = [0, 90, 180, 270]
    flip_direction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE]
    if opened_image_1.size[0] == 400:
        for angle in rotation_angles:
            for flip in flip_direction:
                transformed_image_input = opened_image_1.rotate(angle).transpose(flip)
                transformed_image_target = opened_image_2.rotate(angle).transpose(flip)
                transformed_image_target.convert('1')
                transformed_image_input.save(augmented_input_dir + str(counter).zfill(5) + '.png')
                transformed_image_target.save(augmented_target_dir + str(counter).zfill(5) + '.png')
                counter += 1
                if counter % 100 == 0:
                    print(counter)
    else:
        counter = crop_image(opened_image_1, opened_image_2, size, counter)
    return counter

def load_images(idx, folder = 'train_augmented2/'):
    sat = folder+'input/'
    m = folder+'target/'
    sat_file = sat+str(idx).zfill(5) + '.png'
    map_file = m+str(idx).zfill(5) + '.png'
    
    images = map(Image.open, [sat_file, map_file])
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    # in python3 you can only iterate over a map once
    images = map(Image.open, [sat_file, map_file])
    for img in images:
        #img.show()
        new_img.paste(img, (x_offset,0))
        x_offset += img.size[0]
    return new_img

def plot_images(idx):
    img = load_images(idx)
    img.show()

# point to the correct directories
original_root_dir = 'training'
only_augmentation_data = False #If true only loads the toronto dataset

original_input_dir = original_root_dir + '/images/'
original_target_dir = original_root_dir + '/target/'

input_dir2 = original_root_dir+'/sat_data/'
target_dir2 = original_root_dir+'/map_data/'

augmented_root_dir = 'train_augmented2'
augmented_input_dir = augmented_root_dir + '/input/'
augmented_target_dir = augmented_root_dir + '/target/'

valid = [original_root_dir,
         original_input_dir, original_target_dir,
         input_dir2, target_dir2,
         augmented_root_dir,
         augmented_input_dir, augmented_target_dir]

for name in valid:
    if not os.path.isdir(name):
        os.mkdir(name)
        
# load images
if only_augmentation_data:
    original_input_images = glob.glob(input_dir2 + '*.png')
    original_target_images = glob.glob(target_dir2 + '*.png')
else:
    original_input_images = glob.glob(original_input_dir + '*.png') + glob.glob(input_dir2 + '*.png')
    original_target_images = glob.glob(original_target_dir + '*.png') + glob.glob(target_dir2 + '*.png')
    print("Using all the data!")
# run through all the images, keep input and target zipped
counter = 0
for original_input_image, original_target_image in zip(original_input_images, original_target_images):
    counter = transform_image_combined(original_input_image,
                                       original_target_image,
                                       counter)
print("Total number of images generated:", counter)
# Check our results
# print('Checking our results...")
# plot_images(np.random.randint(0, counter))