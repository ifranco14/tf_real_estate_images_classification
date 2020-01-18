import clahe
import numpy as np
import pandas as pd
import cv2
import os

from joblib import Parallel, delayed

def resize_image(img, img_size):

    dim = (img_size, img_size)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

def show_image(img=None, img_path=None):
    if img is None:
        if img_path is None:
            return False
        img = cv2.imread(img_path)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_clahe(img, img_size=128, channels=3, clip_limit=0.5):
    '''
    Apply Contrast Limited Adaptive Histogram Equalization filter
    '''
    return clahe.clahe(img, (img_size, img_size, channels), clip_limit)


def open_img(img_path, apply_clahe_filter=False):
    img = cv2.imread(img_path)

    if apply_clahe_filter:
        return make_clahe(img)

    return img

def obtain_image_array(img_path, apply_clahe_filter=True, img_size=256,
                       channels=3, norm_factor=1, clip_limit=.5):
    img = open_img(img_path)
    img = resize_image(img, img_size)

    if apply_clahe_filter:
        return make_clahe(img=img, img_size=img_size, channels=channels,
                          clip_limit=clip_limit)

    return img / norm_factor


def obtain_images_dataset(add_external_data=False, save_data=True):
    df = obtain_tags_dataset()

    path = PATH.RAW_DATA_PATH + 'train/'

    arrays = []
    for img in os.listdir(path):
        img_path = f'{path}/{img}'

        try:
            img_array = obtain_image_array(img_path,
                                           apply_clahe_filter=True,
                                           img_size=128,
                                           norm_factor=255)
            arrays.append((img, img_array))
        except Exception as e:
            print(f"Image {img} couldn't be load ({img_path})")
            print(e)
            #continue
            break

    return arrays


def obtain_tags_dataset():
    path = PATH.RAW_DATA_PATH

    train_tags_df_path = f'{path}/train.csv'

    return pd.read_csv(train_tags_df_path)





import numpy as np
import os
import cv2
from PIL import Image

from src.data import data


def apply_clahe_and_save(img_name, dataset_path, dataset, category, processed_data_path, ):
    img_file_path = f'{dataset_path}/{dataset}/{category}/{img_name}'
    new_category_path = f'{processed_data_path}/right_clahe_vision_based_dataset/{dataset}/{category}/'

    if not os.path.exists(new_category_path):
        os.makedirs(new_category_path)
    clahe_img_file_path = f'{new_category_path}/{img_name}'

    # clahe_img = apply_clahe(img_file_path)
    clahe_img = obtain_image_array(img_file_path,
                                   apply_clahe_filter=True,
                                   img_size=256,
                                   norm_factor=255,
                                   channels=3,
                                   clip_limit=2)

    cv2.imwrite(clahe_img_file_path, clahe_img*255)

    return 1

def safe_clahe_conversion(img_name, dataset_path, dataset, category, processed_data_path,):
    try:
        return apply_clahe_and_save(img_name, dataset_path, dataset, category, processed_data_path, )
    except:
        print('Count not convert', img_name, 'of category', category,'in dataset',dataset)
        pass


def main():
    processed_data_path = f'{data.PATH().PROCESSED_DATA_PATH}'
    dataset_path = f'{processed_data_path}/right_vision_based_dataset/'

    datasets = ['train', 'validation', 'test']
    datasets = ['test']
    for dataset in datasets:
        categories = os.listdir(f'{dataset_path}/{dataset}')

        for category in categories[::-1]:
            images = os.listdir(f'{dataset_path}/{dataset}/{category}')
            print(f'Processing {len(images)} of category {category} from {dataset} dataset')

            r = Parallel(n_jobs=-1, verbose=100,)(delayed(safe_clahe_conversion)(
                img_name, dataset_path, dataset, category, processed_data_path,)
                                                  for img_name in images)

if __name__ == '__main__':
    main()
