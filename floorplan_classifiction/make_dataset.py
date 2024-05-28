import os

import PIL
from PIL.Image import Image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import random
import shutil
from convert_pdf_to_jpg import delete_files_in_directory
import cv2


def enh(img,factor,s=(224,224)): # image_enhacer
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(s, Image.LANCZOS)

    enhancer = ImageEnhance.Sharpness(img)
    sharpened_img = enhancer.enhance(factor)  # Adjust the enhancement factor as needed

    return sharpened_img


def apply_clahe(path,clipLimit,save=True):
    image = cv2.imread(path)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(10, 10))
    final_img = clahe.apply(image_bw)
    rgb_image = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)

    if save:
        cv2.imwrite(path, rgb_image)
    else:
        return rgb_image



def sharp_and_res(data_dir = "data",factor=5,clipLimit=7):

    """
    Apply processing techniques on the images in data_dir directory
    """

    for image_type in os.listdir(data_dir):
        type_dir = os.path.join(data_dir, image_type)
        image_files = os.listdir(type_dir)

        for file in image_files:
            try:
                img = Image.open(os.path.join(type_dir, file))
            except PIL.UnidentifiedImageError:
                os.remove(os.path.join(type_dir, file))
                print(f"File '{os.path.join(type_dir, file)}' has been deleted.")
                continue

            img = enh(img,factor)
            img.save(os.path.join(type_dir, file))

            if image_type == 'surveys':
                apply_clahe(os.path.join(type_dir, file),clipLimit) # just give a path


def save_to_dataset(data_dir = "data",dataset_dir = "dataset"):

    """
    Take images from data_dir and distribute them over the Dataset_original directory in the following way:

    10% for testing for all image types
    15% for validation for all image types
    75% for training for all image types
    """

    test_percentage = 0.1
    validation_percentage = 0.15

    for image_type in os.listdir(data_dir):

        type_dir = os.path.join(data_dir, image_type)

        image_files = os.listdir(type_dir)

        train_path = f"{dataset_dir}/train/{image_type}"
        test_path = f"{dataset_dir}/test/{image_type}"
        validation_path = f"{dataset_dir}/validation/{image_type}"

        if not os.path.exists(train_path):
            os.makedirs(train_path)

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        num_images = len(image_files)
        num_test = int(num_images * test_percentage)
        num_validation = int(num_images * validation_percentage)
        num_train = num_images - (num_test + num_validation)

        # Randomly select and move images to the test set
        random.shuffle(image_files)

        test_files = image_files[:num_test]
        validation_files = image_files[num_test:num_test + num_validation]
        train_files = image_files[num_test + num_validation:]

        ##################################################

        for file in test_files:
            shutil.copy(os.path.join(type_dir, file), os.path.join(test_path, file))

        for file in validation_files:
            shutil.copy(os.path.join(type_dir, file), os.path.join(validation_path, file))

        for file in train_files:
            shutil.copy(os.path.join(type_dir, file), os.path.join(train_path, file))




def count_files_in_directory(directory_path): # count images in folder
    file_count = 0
    for entry in os.scandir(directory_path):
        if entry.is_file():
            file_count += 1
    return file_count


def augementation(survey_images_dir,floors_images_dir,output_dir):

    # Create an ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,  # rotate images by 20 degrees
        width_shift_range=0.2,  # shift images horizontally by 20%
        height_shift_range=0.2,  # shift images vertically by 20%
        shear_range=0.2,  # shear transformations
        zoom_range=0.2,  # zoom in/out by 20%
        horizontal_flip=True,  # allow horizontal flipping
        vertical_flip=True  # allow vertical flipping
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # List all survey image filenames
    survey_images = [os.path.join(survey_images_dir, img) for img in os.listdir(survey_images_dir)]

    floors_images_len = count_files_in_directory(floors_images_dir) # the number of images in floor_plans training
    survey_images_len = count_files_in_directory(survey_images_dir) # the number of images in survey training

    # 1000 -> 700 <= x <= 850

    a = floors_images_len*0.7
    b = floors_images_len*0.85
    a = a//survey_images_len
    b = b//survey_images_len
    num = round(random.uniform(a, b))

    print("done Counting\n")
    print(floors_images_len)
    print(num)
    print(survey_images_dir)
    print(floors_images_dir)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Perform augmentation and save the augmented images
    for img_path in survey_images:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        # Reshape the image to (1, height, width, channels) for flow() function
        img = img.reshape((1,) + img.shape)

        # Generate augmented images
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= num:  # generate num augmented images for each original image
                break

    delete_files_in_directory(directory_path=survey_images_dir)

    for image_file in os.listdir(output_dir):
        shutil.copy(os.path.join(output_dir, image_file), os.path.join(survey_images_dir, image_file))

    delete_files_in_directory(directory_path=output_dir)
