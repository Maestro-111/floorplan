
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

from direction_detection import get_direction
import pandas as pd


# Build paths inside the project like this: BASE_DIR / 'subdir'.

BASE_DIR = Path(__file__).resolve().parent.parent

#model = keras.models.load_model(os.path.join(BASE_DIR,'keyplates_classifier.keras'))
model = keras.models.load_model(os.path.join(BASE_DIR,'key_plates_new.keras'))

class_names = ['key_plates', 'other']
DIM = (224,224)

SAVE_LOC = os.path.join(BASE_DIR,'unit_testing/unit_number_direction.xlsx') # where to save unit numbers

UNIT_SHEET = 'Unit Number Info'
DIRECTION_SHEET = 'Direction Info'

PYTHON_PATH = os.path.join(BASE_DIR,'CRAFT/venv/Scripts/python.exe') # python used for CRAFT
UNIT_NUMBER_SAVE_LOC = os.path.join(BASE_DIR,'unit_number_ocr/unit_numbers.txt')

DIRECTION_SAVE_LOC = os.path.join(BASE_DIR,'unit_testing/direction.xlsx')

THRESHOLD = 0.5

CRAFT_PATH = os.path.join(BASE_DIR,"CRAFT/test.py")
CRAFT_MODEL_PATH = os.path.join(BASE_DIR,"CRAFT/craft_mlt_25k.pth")

data = os.path.join(BASE_DIR, 'sample')

IMAGE_NUM = 0
root = 72

cut_pixels = 10 # cut image for all directions by 10 pixels


def clear_sheet(excel_file, sheet_name):

    """
    Clearing excel file in unit testing folder
    """
    # Load the Excel workbook
    wb = openpyxl.load_workbook(excel_file)

    # Get the specified sheet
    sheet = wb[sheet_name]

    # Clear all cells in the sheet
    for row in sheet.iter_rows():
        for cell in row:
            cell.value = None

    # Save the changes
    wb.save(excel_file)





def run_craft():

    """

    Run craft module to extract and save unit numbers in 2 locations:

    1) for unit testing
    2) in this folder to paste unit numbers into the direction template

    """

    command = [PYTHON_PATH, CRAFT_PATH, "--trained_model", CRAFT_MODEL_PATH,
               "--unit_number_save_loc",UNIT_NUMBER_SAVE_LOC,
               '--text_threshold','0.05','--low_text','0.05','--link_threshold','0.5','--key_plates','True','--unit_testing_save_path',f'{SAVE_LOC}']
    subprocess.run(command)

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


def is_contour_inside(contour1, contour2,eps=0): # check if contour1 in contour2; eps - allow small deviation
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    return x2 <= x1+eps and y2 <= y1 + eps and x2 + w2 + eps >= x1 + w1 and y2 + h2 + eps >= y1 + h1

def convert_to_cnt(points_list):
    """
    convert points list to contours
    """
    contours = []
    for points in points_list:
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        contours.append(contour)
    return contours

def remove_contours_inside(contours):
    filtered_contours = []

    # Iterate through each contour
    for i in range(len(contours)):
        is_inside = False

        # Compare with other contours
        for j in range(len(contours)):
            if i != j and is_contour_inside(contours[i], contours[j]):
                is_inside = True
                break

        # If the contour is not inside any other, add it to the result
        if not is_inside:
            filtered_contours.append(contours[i])

    return filtered_contours

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

def distance_between_contours(contour1, contour2):
    # Calculate the distance between closest points of two contours
    min_dist = float('inf')
    for pt1 in contour1:
        for pt2 in contour2:
            dist = np.linalg.norm(pt1 - pt2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def merge_close_contours(contours, threshold=30):
    merged_contours = []
    merged_indices = set()

    for i, contour1 in enumerate(contours):
        if i in merged_indices:
            continue
        for j, contour2 in enumerate(contours):
            if i != j and j not in merged_indices:
                dist = distance_between_contours(contour1, contour2)
                if dist < threshold:
                    # Merge contours
                    merged_contour = np.concatenate((contour1, contour2))
                    merged_contours.append(merged_contour)
                    merged_indices.add(i)
                    merged_indices.add(j)
                    break
        if i not in merged_indices:
            # If no contour was close enough to merge, add the contour as is
            merged_contours.append(contour1)

    return merged_contours

def enhance_and_reshape(img,factor,s): # image_enhacer

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(s, Image.LANCZOS)

    enhancer = ImageEnhance.Sharpness(img)
    sharpened_img = enhancer.enhance(factor)  # Adjust the enhancement factor as needed

    return sharpened_img

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def main():

    global IMAGE_NUM,CONVERSATION_ID

    """

    for each image:
    prepare it for contours extraction,
    extract contours
    classify keyplates among them
    run craft on keyplates, save 2 things: unit number in unit testing folder and unit numbers in txt file in this folder
    aply gpt prompt with saved  unit number in txt file to get directions
    store the directions results in unit testing folder
    """

    clear_sheet(SAVE_LOC,UNIT_SHEET) # make sure excel file is empty
    clear_sheet(SAVE_LOC, DIRECTION_SHEET)

    for initial_image_path in os.listdir(data):

        path = os.path.join(data, initial_image_path)
        full_initial_image_path = os.path.join(data, initial_image_path)

        image_original = cv2.imread(path)
        height, width = image_original.shape[:2]

        new_height = height - 2 * cut_pixels
        new_width = width - 2 * cut_pixels

        image_original = image_original[cut_pixels:new_height + cut_pixels, cut_pixels:new_width + cut_pixels]

        gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)


        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True) # sort by area and flter out the biggest since its the image itself
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

        merged_contours = [cnt for cnt in merged_contours if cv2.contourArea(cnt) > 3000]

        count_images = 0
        count_txt = 0

        # store coords of the boxes and contours to the appropriate folders

        for cnt in merged_contours:

            x, y, w, h = cv2.boundingRect(cnt)
            image_rect = image_original[y:y+h, x:x+w]

            txt_file_path = f'coords/rect_coordinates_{count_txt}.txt'

            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(f'{x},{y},{w},{h}')


            # root is the number for current program run
            # image num is number of image
            #count_images is number of countour of the partilucar image

            # all this to make each image fraction name unique

            cv2.imwrite(f'tmp_rects/tmp_rect_{root}_{IMAGE_NUM}_{count_images}.jpg', image_rect)

            count_images += 1
            count_txt += 1

        IMAGE_NUM += 1

        images_rect = [os.path.join('tmp_rects', image) for image in os.listdir('tmp_rects')]
        coords_rect = [os.path.join('coords', coord) for coord in os.listdir('coords')]

        paths = list(zip(coords_rect,images_rect))

        key_plates = []

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
                    key_plates.append(line+[prediction])

        contour_image = image_original.copy()
        count = 0

        test_folder = 'test'

        # copy key plates to test folder

        for x, y, w, h,p in key_plates:

            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            plate = image_original[y:y+h,x:x+w]

            path = f'{count}_{initial_image_path}'
            path = os.path.join(test_folder,path)

            print("!!!!!!!!")
            print(path)
            print("!!!!!!!!")

            cv2.imwrite(path, plate)
            count+= 1

        run_craft()

        a = input("Stop here, Enter: ")

        plt.imshow(contour_image)
        plt.show()

        delete_files_in_directory("tmp_rects")
        delete_files_in_directory("coords")
        delete_files_in_directory("test")


create_folder_if_not_exists('coords')
create_folder_if_not_exists('tmp_rects')
create_folder_if_not_exists('test')


if __name__ == '__main__':
    main()