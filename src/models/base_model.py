import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

class BaseModel:
    def __init__(self, batch_size, n_classes, epochs, img_size,
                 n_channels, model_name=''):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.epochs = epochs
        self.img_size = img_size
        self.n_channels = n_channels
        self.model_name = model_name
        self.model = None
        self.model_is_compiled = False
        self.image_generator_class_mode = 'categorical' # categorical/binary/sparse/input
        self.image_generator_color_mode = 'rgb' # grayscale/rgb/rgba
        self.SEED = 14

    def compile(self, optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model_is_compiled = True

    def fit(self, class_weight=None, sample_weight=None):
        '''
        Sample of class_weight:
            class_weight = {
                0: 1.,
                1: 1.,
                2: 1.,
                3: 1.,
                4: 1.,
                5: 2.,
            }
        '''
        pass

    @staticmethod
    def get_class_weights(samples):
        return class_weight.compute_class_weight(
                'balanced', np.unique(samples), samples)

    def fit_from_generator(self, path, steps_per_epoch=1000, weighted=False, ):
        assert self.model_is_compiled, 'The model should be compiled before fit'

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            dtype=tf.float32,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            )


        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            dtype=tf.float32,
            )

        train_generator = train_datagen.flow_from_directory(
                f'{path}/train/',
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode=self.image_generator_class_mode,
                color_mode=self.image_generator_color_mode,
                seed=self.SEED)

        validation_generator = validation_datagen.flow_from_directory(
                f'{path}/validation/',
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode=self.image_generator_class_mode,
                color_mode=self.image_generator_color_mode,
                seed=self.SEED)

        class_weights = None
        if weighted:
            class_weights = self.get_class_weights(train_generator.classes)
            print('class_weights', class_weights)

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=int(steps_per_epoch*.4),
            class_weight=class_weights)

    def evaluate(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test, verbose=0)
        pass

    def fit_and_evaluate(self, ):
        pass
