import cv2
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict


from room_floor_area_ocr import read_coords

import pytesseract

import os
import openpyxl



def calculate_dominant_color(image):

    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Flatten the image to a 1D array of RGB values
    pixels = img_array.reshape(-1, 3)



    # Calculate the histogram of colors
    hist = np.histogramdd(pixels, bins=(256, 256, 256), range=((0, 256), (0, 256), (0, 256)))[0]

    # Get the color with the highest frequency
    dominant_color = np.unravel_index(hist.argmax(), hist.shape)

    return dominant_color


def has_colored_background(image, threshold=30):
    # Calculate the dominant color of the image
    dominant_color = calculate_dominant_color(image)
    return sum(dominant_color)/3 <= 230



def analyze_plates(img_txt,unit_testing_save,unit_number_save):


    numbers = defaultdict(list)
    image_name = ''


    for txt_path,image_path in img_txt: # all represent the same image perhaps different key plate

        name = image_path.split('\\')[-1][:-4]
        name = name[2:]

        image_name = name

        img = Image.open(image_path)
        points = read_coords(txt_path)

        for point in points:
            x1,y1,x2,y2,x3,y3,x4,y4 = point

            try:
                fraction = img.crop((x1, y1, x3, y3))
            except ValueError:
                continue

            if has_colored_background(fraction):
                processed_fraction = fraction.convert('L')
                processed_fraction = processed_fraction.resize((28, 28))

                processed_fraction = processed_fraction.filter(ImageFilter.SHARPEN)
                processed_fraction = ImageEnhance.Contrast(processed_fraction).enhance(2.0)

                extracted_number = pytesseract.image_to_string(processed_fraction, config='--psm 7 outputbase digits')
                extracted_number = extracted_number.replace('\n','')

                numbers[name].append(extracted_number)

    unit_nums_for_image = [';'.join(numbers[image_name])]

    with open(unit_number_save,'w') as f:
        f.write(unit_nums_for_image[0])

    names = [image_name]

    matrix = np.array([names, unit_nums_for_image])


    new_df = pd.DataFrame(matrix.transpose(), columns=['Name', 'Unit Number'])

    print(new_df)


    existing_df = pd.read_excel(unit_testing_save)

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    with pd.ExcelWriter(unit_testing_save, mode='a', engine='openpyxl',if_sheet_exists='overlay') as writer:
        combined_df.to_excel(writer, index=False, sheet_name='Unit Number Info')

