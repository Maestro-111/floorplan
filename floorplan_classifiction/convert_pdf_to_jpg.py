import os
from PIL.Image import Image
import fitz
import shutil
import random


def removeFile(filename):
    if os.path.exists(filename):
        os.remove(filename)

def extract_images_from_pdf(pdf_path, output_folder,num):
    imageFiles = []

    pdf_filename = os.path.basename(pdf_path)
    pdf_doc = fitz.open(pdf_path)

    for page_index in range(len(pdf_doc)):

        filepath = f'{output_folder}/pdf_floor_plan_{num}_{page_index}.jpg'

        page = pdf_doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Identity, dpi=250)

        removeFile(filepath)
        pix.save(filepath)  # save file
        imageFiles.append(filepath)

def make_floor_plans(pdf_dir = 'in_pdf',output_folder_path = "floor_plans", prob=0.0):


    num = 0
    print("Copying FloorPlans Images")

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for pdf_file in os.listdir(pdf_dir):
        path = os.path.join(pdf_dir, pdf_file)
        if random.random() > prob:
            extract_images_from_pdf(path, output_folder_path,num)
            num += 1

    print("Done Copying FloorPlans Images")

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


def copy_sur(data_dir = "survey_original",out_dir='data/surveys',prob=0.0):

    """
    we specify prob as prob from 0 to 1. This is to regulte what propotion of images to load. In training we use 0.0 to load all the images
    """

    num = 0
    print("Copying Survey Images")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in os.listdir(data_dir):
        if random.random() > prob:
            new_filename = f'surveys_{num}.jpg'
            new_path = os.path.join(out_dir, new_filename)
            shutil.copy(os.path.join(data_dir, file), new_path)
            num += 1
    print("Done Copying Survey Images")


def copy_other(data_dir = "other_images",out_dir='data/other_images',prob=0.0):

    num = 0
    print("Copying other_images")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in os.listdir(data_dir):
        if random.random() > prob:
            new_filename = f'other_{num}.jpg'
            new_path = os.path.join(out_dir, new_filename)
            shutil.copy(os.path.join(data_dir, file), new_path)
            num += 1
    print("Done Copying other_images")