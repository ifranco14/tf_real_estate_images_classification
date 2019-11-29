from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import initializers, regularizers
from src.models import base_model

class CNN_Model(base_model.BaseModel):
    def __init__(self, batch_size, n_classes, epochs, img_size, n_channels,
                 experiment_name='',version=1):
        super().__init__(batch_size, n_classes, epochs, img_size, n_channels,
                         model_name=type(self).__name__,
                         experiment_name=experiment_name)

        input_shape = (img_size, img_size, n_channels)
        self.model = self.get_model(input_shape=input_shape, version=version)

    def get_model(self, input_shape, version=None):

        if version is None or version == 1:
            return self.get_model_v1(input_shape)
        elif version == 2:
            return self.get_model_v2(input_shape)
        else:
            raise NotImplementedError()

    def get_model_v1(self, input_shape, ):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape,
                         kernel_initializer=initializers.glorot_uniform(seed=self.SEED),
                         bias_initializer=initializers.Constant(0.1)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01),
                        activation='relu',))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model


    def get_model_v2(self, input_shape, ):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape,
                         kernel_initializer=initializers.glorot_uniform(seed=self.SEED),
                         bias_initializer=initializers.Constant(0.1)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model
