import cv2
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

import os

from PIL import Image
from PIL import ImageEnhance

import subprocess

import openpyxl

from pathlib import Path
import os

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DIM = (224,224)

model = keras.models.load_model(os.path.join(BASE_DIR,'directions_classifier.keras'))
data = os.path.join(BASE_DIR, 'sample')

OUTPUT_PATH = os.path.join(BASE_DIR,'direction_coords')

class_names = ['directions', 'other']

THRESHOLD = 0.5

IMAGE_NUM = 0
root = 4


def merge_intersecting_rectangles(rectangles, n_iterations):

    def intersects(rect1, rect2): # rect2 inters rect1

        def intersect(p_left, p_right, q_left, q_right):
            return min(p_right, q_right) > max(p_left, q_left)

        x1, y1, x2, y2, x3, y3, x4, y4 = rect1
        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = rect2

        if y1<=dy1 <= y3 and (x1<=dx1<=x2 or x1<=dx2<=x2):
            return True
        if y1<=dy3 <= y3  and (x1<=dx1<=x2 or x1<=dx2<=x2):
            return True

        return False

    def merge_rectangles(rect1, rect2): # rect1 = x1,y1,x2,y2,x3,y3,x4,y4

        if not rect1:
            return rect2

        #print(rect1,rect2)

        x1_min = min(rect1[0], rect2[0])
        y1_min = min(rect1[1], rect2[1])
        x1_max = max(rect1[4], rect2[4])
        y1_max = max(rect1[5], rect2[5])

        w,h = x1_max-x1_min,y1_max-y1_min

        return [x1_min, y1_min,x1_min+w,y1_min,x1_max, y1_max,x1_min,y1_min+h]

    merged_rectangles = rectangles

    for _ in range(n_iterations):

        processed_indices = set()
        current_merged = []

        for i, rect1 in enumerate(merged_rectangles):
            if i in processed_indices:
                continue
            merged_rect = [rect1]
            merged = False

            for j, rect2 in enumerate(merged_rectangles):
                if intersects(rect1, rect2) and i != j:
                    merged_rect.append(rect2)
                    processed_indices.add(j)
                    merged = True

            figure = []

            for rec in merged_rect:
                figure = merge_rectangles(figure,rec)
            current_merged.append(figure)
        merged_rectangles = current_merged

    return merged_rectangles

def convert_to_cnt(points_list):
    """
    convert points list to contours
    """
    contours = []
    for points in points_list:
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        contours.append(contour)
    return contours

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")



def enhance_and_reshape(img,factor,s): # image_enhacer

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(s, Image.LANCZOS)

    enhancer = ImageEnhance.Sharpness(img)
    sharpened_img = enhancer.enhance(factor)  # Adjust the enhancement factor as needed

    return sharpened_img


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



def main():

    global IMAGE_NUM

    for initial_image_path in os.listdir(data):

        path = os.path.join(data, initial_image_path)
        full_initial_image_path = os.path.join(data, initial_image_path)

        image_original = cv2.imread(path)

        gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        sorted_contours = sorted(contours, key=cv2.contourArea,
                                 reverse=True)  # sort by area and flter out the biggest since its the image itself


        contours = sorted_contours[1:]

        filtered_cnt_by_area = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if 100 <= area <= 35000:
                filtered_cnt_by_area.append(contour)

        rectangles = []

        for contour in filtered_cnt_by_area:
            x,y,w,h = cv2.boundingRect(contour)
            rectangles.append([x,y,x+w,y,x+w,y+h,x,y+h])

        merged_rectangles = merge_intersecting_rectangles(rectangles,n_iterations=10)
        merged_rectangles = convert_to_cnt(merged_rectangles)
        merged_contours = merged_rectangles

        merged_contours = [cnt for cnt in merged_contours]

        # for cnt in merged_contours:
        #     cv2.drawContours(image_original, [cnt], -1, (0, 255, 0), 2)
        #
        # plt.imshow(image_original)
        # plt.show()

        count_images = 0
        count_txt = 0

        # store coords of the boxes and contours to the appropriate folders

        for cnt in merged_contours:

            x, y, w, h = cv2.boundingRect(cnt)
            image_rect = image_original[y:y+h, x:x+w]

            txt_file_path = f'coords/rect_coordinates_{count_txt}.txt'

            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(f'{x},{y},{w},{h}')

            cv2.imwrite(f'tmp_rects/tmp_rect_{root}_{IMAGE_NUM}_{count_images}.jpg', image_rect)

            count_images += 1
            count_txt += 1

        IMAGE_NUM += 1

        images_rect = [os.path.join('tmp_rects', image) for image in os.listdir('tmp_rects')]
        coords_rect = [os.path.join('coords', coord) for coord in os.listdir('coords')]

        paths = list(zip(coords_rect, images_rect))
        directions = []

        contour_image = image_original.copy()

        for coord,path in paths:

            img = Image.open(path)
            img = enhance_and_reshape(img, 7, DIM) # prep
            img_array = np.array(img)
            img = tf.expand_dims(img, axis=0)

            prediction = model.predict(img)[0][0]

            if prediction>THRESHOLD:
                predicted_class_index = 1
            else:
                predicted_class_index = 0

            predicted_class = class_names[predicted_class_index]

            print(prediction)
            print(predicted_class_index)
            print(predicted_class)

            if predicted_class_index == 0: # means keyplate

                with open(coord, 'r') as txt_file:
                    line = txt_file.readlines()
                    line = line[0]
                    line = line.split(',')
                    line = list(map(int,line))
                    directions.append(line+[prediction])

        for x, y, w, h,p in directions:
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            p,suffix = initial_image_path.split(".")
            coords_txt_path = os.path.join(OUTPUT_PATH, f"{p}.txt")

            print(coords_txt_path)

            with open(coords_txt_path, 'w') as f:
                f.write(f"{x},{y},{w},{h}")

        plt.imshow(contour_image)
        plt.show()

        delete_files_in_directory("tmp_rects")
        delete_files_in_directory("coords")
        delete_files_in_directory("test")


create_folder_if_not_exists(OUTPUT_PATH)
create_folder_if_not_exists('tmp_rects')
create_folder_if_not_exists('test')
create_folder_if_not_exists('coords')

if __name__ == "__main__":
    main()

