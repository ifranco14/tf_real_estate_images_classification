import numpy as np
import pandas as pd
import os
import shutil
import cv2

class PATH:
    BASE_PATH = '/home/ifranco/Documents/facultad/tesis/tesis/'

    DATA_PATH = f'{BASE_PATH}/data/'

def load_dataset(dataset='rei', force_creation=False):
    paths = PATH()

    if dataset == 'rei':
        path = f'{paths.DATA_PATH}/external/rei_dataset/'

        dst = f'{path}/imgs/'

        if force_creation or not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)

            c = 0
            y = []
            folders = [f'{path}/{f}'
                       for f in os.listdir(path)
                       if 'README' not in f and f != 'imgs' and 'labels' not in f]
            for idx, folder in enumerate(folders):
                for file in os.listdir(folder):
                    y_value = [idx, folder, c]
                    y.append(y_value)
                    shutil.move(f'{folder}/{file}',
                                f'{dst}/{c}.{file.split(".")[-1]}')
                    c += 1

            df_y = pd.DataFrame(y, columns=['encoded_label', 'label', 'file'])
            df_y.to_csv(f"{path}/labels.csv")
        else:
            df_y = pd.read_csv(f'{path}/labels.csv')

        y_labels = df_y.encoded_label.values
        x = transform_images_to_np_arrays(path=dst)

    return x, y_labels


def transform_images_to_np_arrays(path, target_size=128):
    imgs = []
    for img in os.listdir(path):
        im = cv2.imread(f'{path}/{img}')
        resized = cv2.resize(im, (target_size, target_size))
        imgs.append(resized)
    return np.array(imgs)


def get_image_generator_of_dataset(dataset='rei',
                                   batch_size=32, target_size=128,
                                   class_mode='binary',):
    paths = PATH()

    if dataset == 'rei':
        path = f'{paths.DATA_PATH}/external/rei_dataset/'

    if dataset == '':
        return None


    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode=class_mode)

    return train_generator
