import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import random
import shutil
from convert_pdf_to_jpg import delete_files_in_directory
import matplotlib.pyplot as plt
import re
import cv2
from make_dataset import enh
from make_dataset import apply_clahe


from convert_pdf_to_jpg import make_floor_plans
from convert_pdf_to_jpg import copy_sur
from convert_pdf_to_jpg import copy_other


def make_test_dir(r,floor_path,survey,other,output_folder):
    """
    copy test data to the output_folder

    :param r:
    :param floor_path:
    :param survey:
    :param other:
    :param output_folder:
    :return: None
    """

    make_floor_plans(pdf_dir = floor_path,output_folder_path=output_folder,prob=r)
    copy_sur(data_dir=survey, out_dir=output_folder,prob=r)
    copy_other(data_dir=other, out_dir=output_folder,prob=r)


def preprocess_new_data(file_path, factor, clipLimit, img_height=224, img_width=224):
    """
    Apply the processing technique on the new image

    :param file_path:
    :param factor:
    :param img_height:
    :param img_width:
    :return:
    """
    img = Image.open(file_path)
    img = enh(img, factor, (img_height,img_width))
    img.save(file_path)

    if re.findall(r'[Ss]urvey[sS]_\d+', file_path):
        img = apply_clahe(file_path,clipLimit,save=False) # not None, but image array

    img_array = np.array(img)
    return img_array

