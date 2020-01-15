import pickle
from joblib import wrap_non_picklable_objects
import os
import json
import random
import tensorflow as tf
import numpy as np
from datetime import datetime as dt

from tensorflow import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from sklearn.utils import class_weight
from matplotlib import pyplot as plt

# activations display
from PIL import Image
import numpy as np
from keract import get_activations, display_heatmaps, display_activations
from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array

from src.data import data
from src import custom_losses

def init_seeds(seed_value):

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

class BaseModel:
    def __init__(self, batch_size, n_classes, epochs, img_size,
                 n_channels, model_name='', experiment_name=''):
        # Train params
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.epochs = epochs
        self.img_size = img_size
        self.n_channels = n_channels
        self.image_generator_class_mode = 'categorical' # categorical/binary/sparse/input
        self.image_generator_color_mode = 'rgb' # grayscale/rgb/rgba
        self.SEED = 14
        self.training_time = None

        # Model instance info
        self.model_name = model_name
        self.model = None
        self.model_is_compiled = False
        self.model_is_trained = False
        self.model_path = f'{data.PATH().BASE_PATH}/models/{self.model_name}/'
        self.model_path += f"{experiment_name}_" if experiment_name != '' else ''
        self.model_path += f"{dt.now().strftime('%Y-%m-%d__%H_%M')}/"
        self.model_hist = None
        self.model_summary = None
        self.loss = None
        self.metrics = None
        self.optimizer = None
        self.class_weights = None

    def compile(self, optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy']):
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model_is_compiled = True

    @staticmethod
    def load_model(model_path):
        '''
        Loads an instance of model_class with the trained net in the 'model' attribute
        '''
        assert os.path.exists(model_path), 'Path does not exist'
        tf.keras.backend.clear_session()

        filehandler = open(f'{model_path}/instance.pkl', 'rb')
        model = pickle.load(filehandler, )
        filehandler.close()

        with open(f'{model_path}/model.json', 'r') as f:
            loaded_model_json = f.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)

        model.model = loaded_model

        loss = model.loss if model.class_weights is None \
            and model.loss is None \
            else custom_losses.weighted_categorical_crossentropy(
                model.class_weights)

        model.model.compile(optimizer=model.optimizer, loss=loss,
                            metrics=model.metrics)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess = keras.backend.get_session()
        sess.run(init)

        model.model.load_weights(f"{model_path}/model.h5")

        return model

    def save_model(self, ):
        assert self.model_is_compiled, 'compile model before saving'
        assert self.model_is_trained, 'train model before saving'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(f'{self.model_path}/model.json', "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(f'{self.model_path}/model.h5')

        self.model = None

        print('weights', self.class_weights)
        if self.class_weights is not None:
            self.loss = None

        filehandler = open(f'{self.model_path}/instance.pkl', 'wb')
        pickle.dump(self, filehandler)
        filehandler.close()

        m = self.load_model(self.model_path)
        self.model = m.model

    def show_summary(self,):
        assert self.model_is_compiled, 'compile model before show it'

        print('#' * 5, self.model_name, '#' * 5)
        self.model_summary = self.model.summary()
        print()

    def show_activations(self, dataset_path, generator_class_indices,
                         n_images_per_class=1, layer=None,
                         show_heatmaps=False, subset='train', rescale_img=True):
        """
        Credits to https://github.com/philipperemy/keract
        """

        labels = {v: k for k, v in generator_class_indices.items()}

        images = []
        for class_name in generator_class_indices.keys():
            path = f'{dataset_path}/{subset}/{class_name}/'
            images_in_path = os.listdir(path)
            for n in range(n_images_per_class):
                idx = np.random.choice(np.arange(0, len(images_in_path)))
                image_to_show = images_in_path[idx]
                images.append((class_name, Image.open(f'{path}/{image_to_show}')))

        for class_name, image in images:
            image = image.resize((self.img_size, self.img_size))
            image = img_to_array(image)
            if rescale_img:
                image /= 255

            arr_image = np.array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # image = preprocess_input(image)
            pred_proba = self.model.predict(image)
            label_prediction = np.argmax(pred_proba, axis=1)
            confidence = np.max(pred_proba, axis=1)

            label_prediction = [(labels[l], confidence[idx], pred_proba[idx])
                                for idx, l in enumerate(label_prediction)]

            print(f'y_true: {class_name}. Pred and probs: {label_prediction}')

            activations = get_activations(self.model, image)

            display_activations(activations)

            if show_heatmaps:
                display_heatmaps(activations, arr_image,)

    def show_metrics(self, ):
        assert self.model_is_trained
        history = self.model_hist
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def get_image_data_generator(self, path, train=True, validation=True,
                                 test=False, class_mode_validation=None,
                                 class_mode_test=None):

        result = []

        if train:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                dtype=tf.float32,
                # shear_range=0.2,
                # zoom_range=0.2,
                # horizontal_flip=True,
                # featurewise_center=True,
                # featurewise_std_normalization=True,
                # rotation_range=20,
                # width_shift_range=0.2,
                # height_shift_range=0.2,
             )

            train_generator = train_datagen.flow_from_directory(
                f'{path}/train/',
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode=self.image_generator_class_mode,
                color_mode=self.image_generator_color_mode,
                seed=self.SEED,)

            result.append(train_generator)

        if validation:
            validation_datagen = ImageDataGenerator(
                rescale=1./255,
                dtype=tf.float32,
                )

            validation_generator = validation_datagen.flow_from_directory(
                f'{path}/validation/',
                target_size=(self.img_size, self.img_size),
                shuffle=False,
                batch_size=(1 if class_mode_validation is None
                            else self.batch_size),
                class_mode=class_mode_validation,
                color_mode=self.image_generator_color_mode,
                seed=self.SEED)

            result.append(validation_generator)

        if test:
            test_datagen = ImageDataGenerator(
                rescale=1./255,
                dtype=tf.float32,
                )

            test_generator = test_datagen.flow_from_directory(
                f'{path}/test/',
                target_size=(self.img_size, self.img_size),
                shuffle=False,
                batch_size=1 if class_mode_test is None else self.batch_size,
                class_mode=class_mode_test,
                color_mode=self.image_generator_color_mode,
                seed=self.SEED, )

            result.append(test_generator)

        if len(result) > 1:
            return tuple(result)
        else:
            return result[0]


    @staticmethod
    def get_class_weights(samples, instance=None):
        weights =class_weight.compute_class_weight(
            'balanced', np.unique(samples), samples)

        if instance is not None:
            instance.class_weights = weights

        return weights

    def fit_from_generator(self, path, weighted=False,
                           train_generator=None, validation_generator=None,
                           evaluate_net=False, use_early_stop=True,
                           use_model_check_point=True, log_training=True,
                           n_workers=3, show_activations=False,
                           test_generator=None, initial_epoch=0):
        assert self.model_is_compiled, 'The model should be compiled before fit'

        if train_generator is None:
            train_generator = self.get_image_data_generator(
                path, train=True, validation=False)

        if validation_generator is None:
            validation_generator = self.get_image_data_generator(
                path, train=False, validation=True)

        class_weights = None
        if weighted:
            class_weights = self.get_class_weights(train_generator.classes)
            print('class_weights', class_weights)
            # class_weights = {idx: w for idx, w in enumerate(class_weights)}
            self.class_weights = class_weights
            print('**** Class weights ****')
            print(self.class_weights)
            print('*** ** *** *** *** ** ***')

        callbacks = []

        if log_training:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            csv_logger = CSVLogger(f'{self.model_path}/training.log')
            # csv_logger = None
            callbacks.append(csv_logger)

        if use_early_stop:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='min')
            callbacks.append(early_stop)

        if use_model_check_point:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            check_point = keras.callbacks.ModelCheckpoint(
                f'{self.model_path}/{filepath}', monitor='val_categorical_accuracy', verbose=0,
                save_best_only=True, save_weights_only=True,
                mode='max', period=1)
            callbacks.append(check_point)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess = keras.backend.get_session()
        sess.run(init)

        train_steps_per_epoch = np.ceil(
            len(train_generator.classes) / self.batch_size)
        validation_steps_per_epoch = np.ceil(
            len(validation_generator.classes) / self.batch_size)

        start_training = dt.now()
        self.model_hist = self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=self.epochs,
            verbose=1,
            initial_epoch=initial_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps_per_epoch,
            validation_freq=1,
            max_queue_size=10,
            workers=n_workers,
            use_multiprocessing=True if n_workers > 1 else False,
            shuffle=True,
            class_weight=class_weights,
            callbacks=callbacks if callbacks else None,
        ).history
        self.training_time = (dt.now() - start_training).total_seconds()
        self.model_is_trained = True

        if show_activations:
            self.show_activations(path, train_generator.class_indices,
                                  subset='validation')

        # self.pred = self.predict_from_generator(path, test_generator, validation_generator)
        self.scores = self.evaluate_from_generator(path, validation_generator)
        self.scores_test = self.evaluate_from_generator(path, test_generator)

        self.save_model()

    def evaluate_from_generator(self, path, test_generator=None):
        if test_generator is None:
            test_generator = self.get_image_data_generator(
                path, test=False, train=False, validation=True)

        filenames = test_generator.filenames
        nb_samples = len(filenames)

        # keras.backend.get_session().run(tf.global_variables_initializer())
        print(f'Evaluating generator with {nb_samples} images')
        score = self.model.evaluate_generator(
            test_generator, steps=np.ceil(nb_samples / self.batch_size),
            workers=3,)

        print('Scores:', score)
        print()

        return score

    @staticmethod
    def load_and_evaluate(cls, model_path, dataset_path):
        model = cls.load_model(model_path)

        score = model.evaluate_from_generator(dataset_path,)

        return score

    def predict_from_generator(self, path, test_generator=None,
                               validation_generator=None,
                               return_pred_proba=False):
        if test_generator is None:
            test_generator = self.get_image_data_generator(
                path, test=True, train=False, validation=False)

        if validation_generator is None:
            validation_generator = self.get_image_data_generator(
                path, test=False, train=False, validation=True)


        filenames = test_generator.filenames
        nb_samples = len(filenames)

        # keras.backend.get_session().run(tf.global_variables_initializer())

        pred_proba = self.model.predict_generator(
            test_generator,
            steps=nb_samples,
            workers=3,)

        if return_pred_proba:
            return pred_proba

        labels = {v: k for k, v in validation_generator.class_indices.items()}

        label_prediction = np.argmax(pred_proba, axis=1)
        confidence = np.max(pred_proba, axis=1)

        label_prediction = [(labels[l], confidence[idx])
                            for idx, l in enumerate(label_prediction)]

        return label_prediction

    @staticmethod
    def load_and_predict(cls, model_path, dataset_path):
        model = cls.load_model(model_path)

        return model.predict_from_generator(dataset_path,)
