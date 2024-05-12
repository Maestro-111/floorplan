import subprocess
import shutil
import os

SAVE_LOC = r'C:\floorplan\unit_testing\unit_number.xlsx'

PYTHON_PATH = r'C:/Python39/python.exe'
ocr_save_path = r"C:\floorplan\unit_testing\craft_ocr_output.xlsx"

sample = 'C:/floorplan/sample'


def run_craft():
    python_path = PYTHON_PATH
    command = [python_path, r"C:\CRAFT\test.py", "--trained_model", r"C:\CRAFT\craft_mlt_25k.pth",
               "--custom_prep","True","--ocr_save_path", ocr_save_path]
    subprocess.run(command)

if __name__ == "__main__":

    # copy them

    copy_dir = 'C:/floorplan/room_floor_area_classification/test'

    for image in os.listdir(sample):
        shutil.copy(os.path.join(sample, image), os.path.join(copy_dir,image))

    run_craft()

