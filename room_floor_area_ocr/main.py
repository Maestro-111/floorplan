import subprocess
import shutil

from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent


PYTHON_PATH = os.path.join(BASE_DIR,r'CRAFT\venv\Scripts\python.exe') #'C:/Python39/python.exe'

print(PYTHON_PATH)

area_floor_rooms_save_path = os.path.join(BASE_DIR, "unit_testing/craft_ocr_output.xlsx")

CRAFT_PATH = os.path.join(BASE_DIR,"CRAFT/test.py")
CRAFT_MODEL_PATH = os.path.join(BASE_DIR,"CRAFT/craft_mlt_25k.pth")


data = os.path.join(BASE_DIR, 'sample')


def run_craft():
    command = [PYTHON_PATH, CRAFT_PATH, "--trained_model", CRAFT_MODEL_PATH,
               "--area_floor_rooms","True","--area_floor_rooms_save_path", area_floor_rooms_save_path]
    subprocess.run(command)

if __name__ == "__main__":

    copy_dir = os.path.join(BASE_DIR,'room_floor_area_ocr/test')

    for image in os.listdir(data):
        shutil.copy(os.path.join(data, image), os.path.join(copy_dir,image))

    run_craft()

