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
    MODELS_PATH = f'{BASE_PATH}/models/'


def create_train_validation_test_splits_for_dataset(dataset_name, one_category_test=True,
                                                    test_and_validation_percentage=.2,):

    assert dataset_name.lower().replace(' ', '_').replace('-', '_') \
        in ['rei_dataset', 'vision_based_dataset'], 'DATASET NOT EXISTS'

    paths = PATH()

    all_data_dir = f'{paths.RAW_DATA_PATH}/{dataset_name}/'
    training_data_dir = f'{paths.PROCESSED_DATA_PATH}/{dataset_name}/train/'
    validation_data_dir = f'{paths.PROCESSED_DATA_PATH}/{dataset_name}/validation/'
    testing_data_dir = f'{paths.PROCESSED_DATA_PATH}/{dataset_name}/test/'

    split_dataset_into_test_and_train_sets(
        all_data_dir=all_data_dir, training_data_dir=training_data_dir,
        validation_data_dir=validation_data_dir,
        testing_data_dir=testing_data_dir,
        test_and_validation_data_pct=test_and_validation_percentage,
        one_category_test=one_category_test)


def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir,
                                           validation_data_dir,
                                           testing_data_dir, test_and_validation_data_pct,
                                           one_category_test=True):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=True)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if validation_data_dir.count('/') > 1:
        shutil.rmtree(validation_data_dir, ignore_errors=True)
        os.makedirs(validation_data_dir)
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
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = f'{testing_data_dir}/{"all" if one_category_test else category_name}'
        validation_data_category_dir = validation_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        if not os.path.exists(validation_data_category_dir):
            os.mkdir(validation_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < test_and_validation_data_pct:
                if np.random.choice([0, 1]) == 0:
                    shutil.copy(input_file, validation_data_dir + '/' + category_name + '/' + file)
                else:
                    cat = f"all/{category_name}_" if one_category_test else f"{category_name}/"
                    test_dir = f'{testing_data_dir}/{cat}{file}'
                    shutil.copy(input_file, test_dir)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")


def create_version_of_dataset_for_experiment_two(dataset, percentage_per_class=.3):
    seed = 14
    np.random.seed(seed)

    paths = PATH()
    preprocessed_dataset_path = f'{paths.PROCESSED_DATA_PATH}/{dataset}'

    new_dataset_path = f'{preprocessed_dataset_path}_{int(percentage_per_class * 100)}'

    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    for subset in ['train', ]:
        for category in os.listdir(f'{preprocessed_dataset_path}/{subset}'):
            category_path = f'{preprocessed_dataset_path}/{subset}/{category}'
            files = os.listdir(category_path)

            files_new_dataset = files[:int(len(files)*percentage_per_class)]

            print(f'processing category {category}: all files {len(files)} -'\
                    + f' new version {len(files_new_dataset)}')

            new_category_path = f'{new_dataset_path}/{subset}/{category}/'

            if not os.path.exists(new_category_path):
                os.makedirs(new_category_path)

            for file in files_new_dataset:
                shutil.copy2(f'{category_path}/{file}',
                             f'{new_category_path}/{file}')
