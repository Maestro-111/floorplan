import math
import os
from PIL.Image import Image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import random
import shutil
import cv2
import tensorflow as tf


def augementation(images_dir,desired_per_image,output_dir):

    # Create an ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=5,  # rotate images by 30 degrees
        width_shift_range=0.05,  # shift images horizontally by 30%
        height_shift_range=0.05,  # shift images vertically by 20%
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # List all image filenames
    images = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]


    for img_path in images:
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
            if i >= desired_per_image:  # generate num augmented images for each original image
                break

    delete_files_in_directory(directory_path=images_dir)

    for image_file in os.listdir(output_dir):
        shutil.copy(os.path.join(output_dir, image_file), os.path.join(images_dir, image_file))

    delete_files_in_directory(directory_path=output_dir)


def delete_files_in_directory(directory_path):
    try:
        # List all files in the specified directory
        files = os.listdir(directory_path)
        # Iterate through each file and delete them
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("All files deleted successfully.")


    except Exception as e:
        print(f"An error occurred: {e}")


def save_to_dataset(data_dir = "data",dataset_dir = "dataset"):

    """
    Take images from data_dir and distribute them over the dataset directory in the following way:

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


def enh(img,factor,s=(224,224)): # image_enhacer

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(s, Image.LANCZOS)

    enhancer = ImageEnhance.Sharpness(img)
    sharpened_img = enhancer.enhance(factor)  # Adjust the enhancement factor as needed

    return sharpened_img




def sharp_and_res(data_dir = "dataset",factor=5):

    """
    Apply processing techniques on the images in data_dir directory
    """

    for folder in os.listdir(data_dir):
        path1 = os.path.join(data_dir, folder)

        for image_type in os.listdir(path1):
            path2 = os.path.join(path1, image_type)
            image_files = os.listdir(path2)

            for file in image_files:
                img = Image.open(os.path.join(path2, file))
                img = enh(img,factor)
                img.save(os.path.join(path2, file))


def convert_to_gray(data_dir="dataset"):

    '''
    not using currently

    '''

    for folder1 in os.listdir(data_dir):
        path1 = os.path.join(data_dir, folder1)

        for folder2 in os.listdir(path1):
            path2 = os.path.join(path1, folder2) # dataset/training/number
            for image_path in os.listdir(path2):

                out = os.path.join(path2, image_path)
                image = Image.open(out)

                if image is not None:

                    if image.mode == 'L':
                        continue

                    gray_image  = image.convert('L')
                    gray_image.save(out)


def check_shape(data_dir="dataset"):
    for folder1 in os.listdir(data_dir):
        path1 = os.path.join(data_dir, folder1)

        for folder2 in os.listdir(path1):
            path2 = os.path.join(path1, folder2) # dataset/training/number
            for image_path in os.listdir(path2):

                out = os.path.join(path2, image_path)
                image = Image.open(out)

                if image is not None:
                    print(image.mode)