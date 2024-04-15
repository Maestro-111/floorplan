import numpy as np
import pandas as pd

import cv2
import os
import pytesseract
from matplotlib import pyplot as plt
from collections import defaultdict
from math import factorial

import random

import openai
from openai import OpenAI

import re
import easyocr

PROB = 0.15


def is_contour_inside(contour1, contour2,eps=10): # check if contour1 in contour2; eps - allow small deviation
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    return x2 <= x1+eps and y2 <= y1 + eps and x2 + w2 + eps >= x1 + w1 and y2 + h2 + eps >= y1 + h1

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

def read_coords(txt_file):

    f = open(txt_file, "r")

    points = []

    for line in f:
        line = line.replace('\n', '')
        if line:
            line = list(map(int,line.split(',')))
            points.append(line)

    return points

def distance_between_contours(contour1, contour2):
    # Calculate the distance between closest points of two contours
    min_dist = float('inf')
    for pt1 in contour1:
        for pt2 in contour2:
            dist = np.linalg.norm(pt1 - pt2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def merge_close_contours(contours, threshold):
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


def get_boxes(merged,copy,img,lst,global_mark:int):

    count = 0

    for cnt in merged: # for each box in current image
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_c = img[y:y + h, x:x + w]

        # global mark is used to save each box with its image label

        image_filename = f"roi_{global_mark}_{count}.jpg"
        image_path = os.path.join('contours', image_filename)
        cv2.imwrite(image_path, roi_c)
        count+= 1

        new_width = 4 * roi_c.shape[1]
        new_height = 4 * roi_c.shape[0]

        try:
            resized_roi = cv2.resize(roi_c, (new_width, new_height))
            lst.append(resized_roi)
        except cv2.error:
            lst.append(roi_c)

    #plt.imshow(copy)
    #plt.show()


    for file in os.listdir('contours'):
        file_path = os.path.join('contours', file)
        os.remove(file_path)

    return


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
        merged = False
        processed_indices = set()
        current_merged = []
        for i, rect1 in enumerate(merged_rectangles):
            if i in processed_indices:
                continue
            merged_rect = [rect1]
            for j, rect2 in enumerate(merged_rectangles[i + 1:]):
                if intersects(rect1, rect2):
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
    contours = []
    for points in points_list:
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        contours.append(contour)
    return contours


def visualize_cnts(copy,merged_rectangles):
    for contour in merged_rectangles:
        x1, y1, w, h = cv2.boundingRect(contour)
        x3, y3 = x1+w, y1+h
        cv2.rectangle(copy, (x1, y1), (x3, y3), (0, 255, 0), 2)

    plt.imshow(copy)
    plt.show()


def merge_existing_boxes(path,points,mark):
    img = cv2.imread(path)

    copy = img.copy()


    merged_rectangles = merge_intersecting_rectangles(points, 7) # lst of points, 7 iters
    merged_rectangles = convert_to_cnt(merged_rectangles) # convert to cv2 cnts



    merged_rectangles = merge_close_contours(merged_rectangles,30) # merge 2 times with proximuty = 30 and 10
    merged_rectangles = remove_contours_inside(merged_rectangles)


    merged_rectangles = merge_close_contours(merged_rectangles,10)
    merged_rectangles = remove_contours_inside(merged_rectangles)

    # convert to points again, merge overlaping and convert to cnts afterwards

    ROI = []

    get_boxes(merged_rectangles, copy, img, ROI,mark)

    return ROI



def process_text_from_tesseract(text_data):

    text_data = re.split(r'\n', text_data)
    text_data = ' '.join(text_data)

    return text_data



def run(info_dir,tes_mode:str=4):

    txts_path = []
    images_path = []

    for folder in os.listdir(info_dir):
        path = os.path.join(info_dir,folder)
        for content in os.listdir(path):
            if folder == 'coords':
                txts_path.append(os.path.join(path,content))
            else:
                images_path.append(os.path.join(path,content))

    img_txt = list(zip(txts_path,images_path))

    mark = 1

    data_names = []
    data_inners = []
    data_outers = []
    ocr_resp = []

    for txt_path,image_path in img_txt:
        points = read_coords(txt_path)

        ROI = merge_existing_boxes(image_path,points,mark)

        pattern_sq = r'[sS][qQ]'
        pattern_ft = r'[fF][tT]'

        area_cand = []
        all_sent = []

        reader = easyocr.Reader(['en'])

        for roi in ROI: # for eeach box

            result = reader.readtext(roi)

            sentence = []

            for (bbox, text_out, prob) in result: # check its text

                #print(f" text : {text_out}, prob {prob}")

                if prob > PROB:
                    sentence.append(text_out)

            sentence = ' '.join(sentence)
            all_sent.append(sentence)

        strings = " ; ".join(area_cand)
        all_sent = ' ; '.join(all_sent)

        data_names.append(image_path.split('\\')[-1][:-4])
        ocr_resp.append(all_sent)


    matrix = np.array([data_names,ocr_resp])
    df = pd.DataFrame(matrix.transpose(),columns=['Name', 'Response'])
    df.to_excel("unit_testing\craft_ocr_output.xlsx")




if __name__ == '__main__':
    run('tets_boxes_from_craft')