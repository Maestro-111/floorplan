import os
from convert_pdf_to_jpg import delete_files_in_directory


def clear_dirs():
    delete_files_in_directory(directory_path='data/floor_plans')
    delete_files_in_directory(directory_path='data/surveys')
    delete_files_in_directory(directory_path='data/other_images')

    for type_dir in ['train', 'test', 'validation']:
        for type_image in ['floor_plans', 'surveys','other_images']:
            path = f'dataset/{type_dir}/{type_image}'
            delete_files_in_directory(directory_path=path)


def make_up_dirs():
    os.makedirs('data/floor_plans')
    os.makedirs('data/surveys')
    os.makedirs('data/other_images')
    os.makedirs('surveys')

    for type_dir in ['train', 'test', 'validation']:
        for type_image in ['floor_plans', 'surveys','other_images']:
            path = f'Dataset_original/{type_dir}/{type_image}'
            os.makedirs(path)


if __name__ == "__main__":
    clear_dirs()
    #make_up_dirs()
