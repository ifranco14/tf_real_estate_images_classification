import numpy as np
import pandas as pd
import os
import shutil
import cv2

class PATH:
    BASE_PATH = '/home/ifranco/Documents/facultad/tesis/tf_real_estate_images_classification/'
    DATA_PATH = f'{BASE_PATH}/data/'
    RAW_DATA_PATH = f'{DATA_PATH}/raw/'
    PROCESSED_DATA_PATH = f'{DATA_PATH}/processed/'


def create_train_validation_test_splits_for_dataset(dataset_name):

    assert dataset_name.lower().replace(' ', '_').replace('-', '_') \
        in ['rei_dataset', 'vision_based_dataset'], 'DATASET NOT EXISTS'

    paths = PATH()

    all_data_dir = f'{paths.RAW_DATA_PATH}/{dataset_name}/'
    training_data_dir = f'{paths.PROCESSED_DATA_PATH}/{dataset_name}/train/'
    validation_data_dir = f'{paths.PROCESSED_DATA_PATH}/{dataset_name}/validation/'

    validation_percentage = 0.2
    split_dataset_into_test_and_train_sets(all_data_dir=all_data_dir,
                                           training_data_dir=training_data_dir,
                                           testing_data_dir=validation_data_dir,
                                           testing_data_pct=validation_percentage)


def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir,
                                           testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=True)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=True)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete testing data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")


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
