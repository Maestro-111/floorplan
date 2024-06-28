import os.path

from data_prep import save_to_dataset
from data_prep import delete_files_in_directory
from data_prep import convert_to_gray
from data_prep import augementation
from data_prep import sharp_and_res
from data_prep import check_shape

import keras_tuner
from custom_CNN import create_dataset
from custom_CNN import make_confusion_matrix
from custom_CNN import neural_net_mixin
from custom_CNN import CNN

import numpy as np
import tensorflow as tf
import keras

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline


TARGET_NAME = 'key_plates'
OPPOSITE_NAME = 'other'

CLASS_NAMES = [TARGET_NAME,OPPOSITE_NAME]

DIM = (224,224,3)
FACTOR = 7
SOURCE = 'C:\metadata_craft1'
MODEL_NAME = 'key_plates_new'
MODEL_SUFIX = "h5"

EPOCHS = 100
GRID_EPOCHS = 10
BATCH = 32

dataset_loc = 'dataset'
surveys = 'surveys'


def process_data(source:str,aug:bool,factor=6):

    save_to_dataset(data_dir=source, dataset_dir=dataset_loc)
    sharp_and_res(data_dir = dataset_loc, factor=factor)

    if aug:
        augementation(f'dataset/train/{TARGET_NAME}', 3, surveys)
        augementation(f'dataset/validation/{TARGET_NAME}', 3, surveys)

    print("Dataset has been created\n")


def dataset(train_path,val_path,test_path,dim:tuple,color_mode:str,batch:int):
    train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_dataset(train_path,val_path,test_path,dim,color_mode,batch_size=batch)

    print(class_names)
    print(num_classes)

    return train_dataset, validation_dataset, test_dataset, class_names, num_classes


def delete_data():

    for type_dir in ['train', 'test', 'validation']:
        path = os.path.join(dataset_loc,type_dir)

        if not os.path.exists(path):
            continue

        for folder in os.listdir(path):

            if folder not in CLASS_NAMES:
                os.rmdir(os.path.join(path, folder))
            else:
                delete_files_in_directory(directory_path=os.path.join(path,folder))


def pipeline(delete=False,process=False,aug=False,train_test=False):

    """
    execute training/testing
    """

    if delete:
        delete_data()
    if process:
        process_data(source=SOURCE,aug=aug,factor=FACTOR)

    if train_test:

        if not DIM or len(DIM) < 3:
            raise ValueError

        width,height,length = DIM

        if length == 1:
            color_mode = 'grayscale'
        else: # 3
            color_mode = 'rgb'

        train_dataset, validation_dataset, test_dataset, class_names, num_classes = dataset(
            f"{dataset_loc}/train",
            f"{dataset_loc}/validation",
            f"{dataset_loc}/test",
            (width,height),
            color_mode,
            BATCH)

        train_test_model_and_save(train_dataset, validation_dataset, test_dataset, class_names, num_classes)


def train_test_model_and_save(train_dataset, validation_dataset, test_dataset,class_names, num_classes):

    """
    train and eval CNN net

    """

    CNN_net = CNN(num_classes, DIM, TARGET_NAME)

    #
    # tuner = keras_tuner.RandomSearch(
    #     hypermodel=CNN_net.cnn_tuner,
    #     objective="val_accuracy",
    #     max_trials=8,
    #     executions_per_trial=4,
    #     overwrite=True,
    # )
    #
    # print(tuner.search_space_summary())
    #
    # tuner.search(train_dataset, epochs=GRID_EPOCHS, validation_data=validation_dataset)
    #
    # print(tuner.results_summary())
    #
    # best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    # best_model = tuner.get_best_models(num_models=1)[0]
    #
    # print("Best Hyperparameters:")
    # print(best_hyperparameters.values)

    best_model_for_training = CNN(num_classes, DIM, TARGET_NAME)
    best_model_for_training.reserve_model()

    #best_model_for_training.cnn_tuner(best_hyperparameters,save_model=True)

    history = best_model_for_training.train(train_dataset,validation_dataset,batch_size=BATCH,epochs=EPOCHS)

    best_model_for_training.summary()
    best_model_for_training.plot_training_hist(history, '3-layers CNN', ['red', 'orange'], ['blue', 'green'])
    best_model_for_training.evaluate_model(test_dataset,class_names)

    best_model_for_training.save(f'C:/floorplan/{MODEL_NAME}.{MODEL_SUFIX}')


pipeline(delete=False, process=False, aug=False, train_test=True)
