
import os
import tensorflow as tf
from tensorflow import keras

from keras import regularizers
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Rescaling
from keras import layers
from keras import models
from keras.layers import Activation
#from keras.activations import sigmoid

from keras.optimizers import SGD

import keras_tuner
from keras.models import clone_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image_dataset_from_directory
from keras.layers import BatchNormalization
from sklearn.metrics import recall_score


def create_dataset(directory_train, directory_valid, directory_test, dim, color_mode,batch_size=32, random_seed=42):
    """
    Fuction, to create datasests, from the directory (datasets) we just created
    """

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory_train,
        color_mode = color_mode,
        seed=random_seed,
        image_size=dim,
        batch_size=batch_size)

    sample_images, _ = next(iter(train_dataset))

    # Print the shape of one image
    print("Shape of a sample image:", sample_images[0].shape)

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory_valid,
        color_mode=color_mode,
        seed=random_seed,
        image_size=dim,
        batch_size=batch_size)

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory_test,
        color_mode=color_mode,
        image_size=dim,
        batch_size=batch_size)

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    train_dataset = train_dataset.unbatch()
    validation_dataset = validation_dataset.unbatch()
    test_dataset = test_dataset.unbatch()

    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
    validation_dataset = validation_dataset.batch(batch_size=batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=True)

    return train_dataset, validation_dataset, test_dataset, class_names, num_classes




def make_confusion_matrix(cm, percent=False, categories=None, cmap=plt.cm.Blues, threshold=None, low_color='green', high_color='orange'):
    fig, ax = plt.subplots(figsize=(8, 6))

    if percent:
        cm = np.round(100 * cm / cm.sum(), 2)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im)

    if categories is not None:
        tick_marks = np.arange(len(categories))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_yticklabels(categories)

    for i in range(len(categories)):
        for j in range(len(categories)):
            # Example: Change text color based on a threshold
            if threshold is not None and cm[i, j] < threshold:
                text_color = low_color
            else:
                text_color = high_color
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=text_color)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


class neural_net_mixin:

    """
    mutual methods for neural nets
    """

    @staticmethod
    def plot_learning_curves(history):
        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    @staticmethod
    def plot_training_hist(hist, model_name: str, accuracy_colors, loss_colors):
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.plot(hist.history['accuracy'], color=accuracy_colors[0])
        ax1.plot(hist.history['val_accuracy'], color=accuracy_colors[1])
        ax1.legend(['train acc', 'validation acc'], loc='upper left')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')

        ax2 = ax1.twinx()
        ax2.plot(hist.history['loss'], color=loss_colors[0])
        ax2.plot(hist.history['val_loss'], color=loss_colors[1])
        ax2.legend(['train loss', 'validation loss'], loc='upper right')
        ax2.set_ylabel('loss')

        plt.title(f'{model_name} training accuracy and loss per epoch')
        plt.show()

    @staticmethod
    def plot_cm(labels, predictions, categories):
        cm = confusion_matrix(labels, predictions)
        make_confusion_matrix(cm, percent=False, categories=categories)


    def save(self,model_save_path):
        self.model.save(model_save_path)
        print(f"Model saved at: {model_save_path}")


class CNN(neural_net_mixin):

    """
    CNN neural net
    """

    def __init__(self,num_classes:int,input_shape:tuple,recall_param_name:str):

        self.recall_param_name = recall_param_name

        self.num_classes = num_classes
        self.input_shape = input_shape

    def cnn_tuner(self,hp,save_model=False):

        img_input = layers.Input(shape=self.input_shape)

        x = layers.Rescaling(1.0 / 255)(img_input)

        # Block 1
        x = Conv2D(
            hp.Int('filters_1', min_value=8, max_value=32, step=8),
            (3, 3),
            padding='same',
            name='block1_conv1'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(x)
        x = Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.7, step=0.1))(x)

        x = Conv2D(
            hp.Int('filters_2', min_value=8, max_value=32, step=8),
            (3, 3),
            padding='same',
            name='block1_conv2'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2')(x)
        x = Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.7, step=0.1))(x)

        x = Conv2D(
            hp.Int('filters_3', min_value=8, max_value=32, step=8),
            (2, 2),
            padding='same',
            name='block2_conv1'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool3')(x)
        x = Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.7, step=0.1))(x)

        # regular

        x = Flatten(name='flatten')(x)

        x = Dense(
            units=hp.Int('units_fc1', min_value=8, max_value=64, step=8),
            name='fc1'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(hp.Float('dropout_fc1', min_value=0.1, max_value=0.7, step=0.1))(x)

        x = Dense(
            units=hp.Int('units_fc2', min_value=8, max_value=64, step=8),
            name='fc2'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(hp.Float('dropout_fc2', min_value=0.1, max_value=0.7, step=0.1))(x)

        x = Dense(
            units=hp.Int('units_fc3', min_value=8, max_value=64, step=8),
            name='fc3'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(hp.Float('dropout_fc3', min_value=0.1, max_value=0.7, step=0.1))(x)


        x = Dense(1, activation='sigmoid', name='predictions')(x)
        model = models.Model(img_input, x, name='custom_model')

        loss_func = tf.losses.BinaryCrossentropy()

        learning_rate = hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])  # Define your choices

        model.compile(
            optimizer=SGD(learning_rate=learning_rate),
            loss=loss_func,
            metrics=['accuracy']
        )

        if save_model:
            self.model = model

        return model


    def reserve_model(self):

        """
        holds the same architecture as cnn_tuner. Used in case we want to ignore hyper search and use 'raw' model.
        """

        img_input = layers.Input(shape=self.input_shape)

        x = layers.Rescaling(1.0 / 255)(img_input)

        # Block 1
        x = Conv2D(
            8,
            (3, 3),
            padding='same',
            name='block1_conv1'

        )(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(x)
        x = Dropout(0.4)(x)

        x = Conv2D(
            8,
            (3, 3),
            padding='same',
            name='block1_conv2'
        )(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2')(x)
        x = Dropout(0.3)(x)


        x = Conv2D(
            8,
            (2, 2),
            padding='same',
            name='block1_conv3'
        )(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool3')(x)
        x = Dropout(0.1)(x)

        # regular

        x = Flatten(name='flatten')(x)

        x = Dense(
            units=16,
            name='fc1'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(
            units=40,
            name='fc2'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(
            units=32,
            name='fc3'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x)

        x = Dense(1, activation='sigmoid', name='predictions')(x)

        model = models.Model(img_input, x, name='custom_model')

        loss_func = tf.losses.BinaryCrossentropy()

        model.compile(
            optimizer=SGD(learning_rate=0.01),
            loss=loss_func,
            metrics=['accuracy']
        )

        self.model = model

    def train(self, train, validation, batch_size=15, epochs=10):
        print("Started Training")

        h = self.model.fit(train, validation_data=validation, batch_size=batch_size, epochs=epochs)

        print("Done Training")

        return h

    def summary(self):
        print(self.model.summary())

    def evaluate_model(self, test_dataset, class_names):

        test_loss, test_acc = self.model.evaluate(test_dataset)

        print(f'test accuracy : {test_acc}')
        print(f'test loss : {test_loss}')

        # Confusion matrix
        y_true = np.array([])
        y_pred = np.array([])

        for X, y in test_dataset:
            y_part = self.model.predict(X)  # Get probabilities
            threshold = 0.5  # Adjust threshold if needed
            print(y_part)
            y_pred_class = np.where(y_part > threshold, 1, 0).flatten()  # Convert to class labels (optional)

            y_pred = np.concatenate([y_pred, y_pred_class])  # Concatenate probabilities or class labels
            y_true = np.concatenate([y_true, y])


            y_true_batch_names = [class_names[int(y)] for y in y]
            y_pred_batch_names = [class_names[int(y)] for y in y_pred_class]  # Use class labels if converted

            # Display actual and predicted class names for the current batch
            for i in range(len(y_true_batch_names)):
                actual_name = y_true_batch_names[i]
                predicted_name = y_pred_batch_names[i]
                print(f"Actual: {actual_name}, Predicted: {predicted_name}")
                print()

        target_index = class_names.index(self.recall_param_name)
        recall_target = recall_score(y_true, y_pred, labels=[target_index], average=None)[0]

        test_f1 = f1_score(y_true, y_pred, average="macro")

        print(f'test f1-score : {test_f1}')
        print(f'test recall score for {self.recall_param_name}: {recall_target}')
        self.plot_cm(y_true, y_pred, class_names)  # Assuming plot_cm works with class labels (0 or 1)
        plt.show()
