"""
Add folder creation
"""



import os
import re

from sample import create_datasets
from sample import make_confusion_matrix
from sample import neural_net_mixin
from sample import CNN

from convert_pdf_to_jpg import delete_files_in_directory
from other_prediction import preprocess_new_data
from other_prediction import make_test_dir

from convert_pdf_to_jpg import delete_files_in_directory
from convert_pdf_to_jpg import make_floor_plans
from convert_pdf_to_jpg import extract_images_from_pdf
from convert_pdf_to_jpg import removeFile
from convert_pdf_to_jpg import copy_sur
from convert_pdf_to_jpg import copy_other

from make_dataset import augementation
from make_dataset import count_files_in_directory
from make_dataset import save_to_dataset
from make_dataset import enh
from make_dataset import sharp_and_res
from make_dataset import apply_clahe

from delete_all import clear_dirs
from delete_all import make_up_dirs

import numpy as np
import tensorflow as tf
from tensorflow import keras

import argparse

FACTOR = 5 # factor for image enhance - more it is, more sharppened the image is going to be
LIMIT = 7 # limit for CLAHE prepocessing for survey images - more it is the more drastic changes are going to be applied
HEIGHT = 224
WIDTH = 224



class CustomException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class DataHolder:

    """
    Holder class to keep the names of the va path variables during the program run
    """

    def __init__(self):
        self.floor_plan_train = None
        self.survey_train = None
        self.other_train = None
        self.floor_plan_test = None
        self.survey_test = None
        self.other_test = None




def load(floor, survey, other):

    """
    Copy images from the specified sources to destination
    """



    copy_other(data_dir=other, out_dir='data/other_images')
    copy_sur(data_dir=survey, out_dir='data/surveys') # copy survey images
    make_floor_plans(pdf_dir = floor, output_folder_path='data/floor_plans') # extarct and copy pdf images

    print("done loading")



def preprocess(factor,clipLimit):

    """

    apply preprocessing -> divide into datasets -> augementation on surveys -> save

    """

    sharp_and_res(factor=factor,clipLimit=clipLimit) # apply sharpening

    print("Transformed Images")

    save_to_dataset() # divide to train/test/val


    print("Made dataset")

    # apply augemenation on each set, for each survey image

    augementation('dataset/train/surveys',
                  'dataset/train/floor_plans', 'surveys')

    augementation('dataset/validation/surveys',
                  'dataset/validation/floor_plans', 'surveys')

    print('Done augementation')


def create_data(train_path,val_path,test_path,height,width):

    train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_datasets(train_path, val_path, test_path,
                                                                                                img_height=height, img_width=width)

    print(class_names)
    print(num_classes)

    return train_dataset, validation_dataset, test_dataset, class_names, num_classes




def train_model_and_save(train_dataset, validation_dataset, test_dataset,class_names, num_classes):

    """
    train and eval CNN net

    """


    CNN_net = CNN(num_classes)
    history = CNN_net.train(train_dataset,validation_dataset,epochs=10)
    CNN_net.plot_training_hist(history, '3-layers CNN', ['red', 'orange'], ['blue', 'green'])
    CNN_net.evaluate_model(test_dataset,class_names)
    CNN_net.save('floorplan_classify.keras')


def make_preds(class_names,model_path,path_for_floor_pdfs,path_for_survey_images,path_for_other_images,factor,clipLimit,prob,data_directory = 'test_data'):

    """
    make predictons on the new data

    :param class_names: load from the txt file (modified during train)
    :param model_path:
    :param path_for_floor_pdfs:
    :param path_for_survey_images:
    :param path_for_other_images:
    :param factor: factor to image enhancer
    :param prob: what % of images to include for evaluation. if 0.2 for exampple, we include 1-0.2 = 0.8 % of each image type
    :param data_directory: tets dara dir
    :return:
    """

    loaded_model = keras.models.load_model(model_path)

    print(loaded_model)


    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    delete_files_in_directory('test_data') # load all images to test_data folder
    make_test_dir(prob,path_for_floor_pdfs,path_for_survey_images,path_for_other_images,output_folder=data_directory)


    file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]

    # Make predictions
    predictions = []
    for file_path in file_paths:
        img = preprocess_new_data(file_path,factor, clipLimit, 224,224)
        img = tf.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction using the loaded model
        prediction = loaded_model.predict(img)

        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        predictions.append(predicted_class)


    overall = 0
    correct = 0

    for file_path, prediction in zip(file_paths, predictions):
        print(file_path)
        if re.findall(r'[Ff]loor\s?_?\+?\-?[Pp]lan[s]?',file_path):
            label = 'floor_plans'
        elif re.findall(r'[Ss]urvey[sS]_\d+', file_path):
            label= 'surveys'
        else:
            label = 'other_images'

        if label == prediction:
            correct += 1
        print(f"File: {file_path}, Label: {label} , Prediction: {prediction}")
        overall += 1

    print(correct / overall)
    print(f'Accuracy: {round(correct / overall, 3)}')






def load_dataset_train_preds_chain(load_flag:bool,prep_flag:bool,train_test_flag:bool,preds_flag:bool,holder):

    """
    Execute the workflow according to the given boolean params
    """

    if load_flag:
        floor_plan,survey,other = holder.floor_plan_train, holder.survey_train, holder.other_train
        load(floor_plan,survey,other)

    if prep_flag:
        preprocess(factor=FACTOR,clipLimit=LIMIT)

    if train_test_flag:

        train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_data(
            "dataset/train",
            "dataset/validation",
            "dataset/test",
            height=HEIGHT,width=WIDTH)


        with open('class_names.txt', 'w') as file: #write class names in a file
            file.write("floor_plans\n")
            file.write("other_images\n")
            file.write("surveys")

        train_model_and_save(train_dataset, validation_dataset, test_dataset, class_names, num_classes)

    if preds_flag:

        with open('class_names.txt', 'r') as file:
            class_names = [name.strip('\n') for name in file.readlines()]

        floor_plan, survey, other = holder.floor_plan_test, holder.survey_test, holder.other_test

        print(floor_plan)
        print(survey)
        print(other)

        make_preds(class_names, model_path='floorplan_classify.keras', path_for_floor_pdfs=floor_plan,
                   path_for_survey_images=survey, path_for_other_images=other, factor=FACTOR,clipLimit=LIMIT,
                   prob=0.5)


def main(train_save:int, test:int, training_image_data:str, testing_image_data:str):

    """
    difine the way we will run load_dataset_train_preds_chain
    """

    if train_save and test:
        print("Executing full pipeline\n")
        strategy = 0
    elif not train_save and test:
        print("Executing testing\n")
        strategy = 1
    elif train_save and not test:
        print("Executing training\n")
        strategy = 2
    else:
        print("No actions are required\n")
        exit(0)

    holder = DataHolder() # to hold paths for data during the run

    if strategy in [0,2] and (not training_image_data): # if we want to do training and do not have the path for training data
        raise CustomException("Unspecified env var for training\n")

    if strategy != 1: # in case we are doing training set the names of training paths into holder object
        root = training_image_data

        for image_folder_type in os.listdir(root):
            if image_folder_type == 'floorplan':
                holder.floor_plan_train = os.path.join(root, image_folder_type)
            elif image_folder_type == 'survey':
                holder.survey_train = os.path.join(root, image_folder_type)
            else: # other images
                holder.other_train = os.path.join(root, image_folder_type)

    if strategy in [0,1] and (not testing_image_data):
        raise CustomException("Unspecified env var for testing\n")

    if strategy != 2: #in case we are doing testing save the data paths in holder

        root = testing_image_data

        for image_folder_type in os.listdir(root):
            if image_folder_type == 'floorplan':
                holder.floor_plan_test = os.path.join(root, image_folder_type)
            elif image_folder_type == 'survey':
                holder.survey_test = os.path.join(root, image_folder_type)
            else: # other images
                holder.other_test = os.path.join(root, image_folder_type)



    if strategy == 0: # full pipeline

        load_dataset_train_preds_chain(load_flag=True,prep_flag=True,train_test_flag=True,preds_flag=True,holder=holder)

    elif strategy == 1: # only preds
        load_dataset_train_preds_chain(load_flag=False, prep_flag=False, train_test_flag=False, preds_flag=True,holder=holder)

    elif strategy == 2: # just train
        load_dataset_train_preds_chain(load_flag=True, prep_flag=True, train_test_flag=True, preds_flag=False,holder=holder)

    else: # train and process without data loading
        raise CustomException("Strategy is unknown\n")

    clear_dirs()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_save", type=int, choices=[0, 1],
                        help='load and prep data, then save the trained model', default=1)
    parser.add_argument("--test", type=int, choices=[0, 1], help='predict for new images', default=1)

    parser.add_argument("--image_training_data", help='name of the training directory',
                        default=os.getenv('image_training_data'))  # def - env var

    parser.add_argument("--image_testing_data", help='name of the testing directory',
                        default=os.getenv('image_testing_data'))

    args = parser.parse_args()

    main(args.train_save, args.test, args.image_training_data, args.image_testing_data)



