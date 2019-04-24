from PIL import Image
import random
import glob


def transform_image_combined(image_1, image_2, counter):
    '''
    :param image_1:
    :param image_2:
    :return:
    '''

    opened_image_1 = Image.open(image_1)
    opened_image_2 = Image.open(image_2)

    rotation_angles = [0, 90, 180, 270]
    flip_direction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE]
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

    return counter


# point to the correct directories
original_root_dir = 'training'
original_input_dir = original_root_dir + '/images/'
original_target_dir = original_root_dir + '/groundtruth/'

augmented_root_dir = 'train_augmented'
augmented_input_dir = augmented_root_dir + '/input/'
augmented_target_dir = augmented_root_dir + '/target/'

# load images
original_input_images = glob.glob(original_input_dir + '*.png')
original_target_images = glob.glob(original_target_dir + '*.png')

counter = 0
# run through all the images, keep input and target zipped
for original_input_image, original_target_image in zip(original_input_images, original_target_images):
    counter = transform_image_combined(original_input_image,
                                       original_target_image,
                                       counter)
