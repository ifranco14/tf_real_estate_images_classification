from keras import initializers, regularizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model, Sequential

from src.models import base_model
from src.models.keras_VGG16_places365.vgg16_places_365 import VGG16_Places365


class PlacesOntop_Model(base_model.BaseModel):
    def __init__(self, batch_size, n_classes, epochs, img_size, n_channels,):
        if img_size != 224:
            img_size = 224
            print(f'Warning!! The images will be resized to 224 instead of {img_size}',
                  'because the pretrained model uses that shape')
        super().__init__(batch_size, n_classes, epochs, img_size, n_channels,
                         model_name=type(self).__name__)

        input_shape = (img_size, img_size, n_channels)
        self.model = self.get_model(input_shape=input_shape)

    def get_model(self, input_shape, ):

        model = Sequential()

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape)

        for l in vgg16_places.layers[:11]:
            model.add(l)

        model.add(vgg16_places.layers[-1])

        for l in model.layers:
            l.trainable = False

        #x = vgg16_places.output
        # x = Flatten(name='flatten')(x)
        model.add(Flatten(name='flatten'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax', name="predictions"))
        # x = Dense(4096, activation='relu',
                  # kernel_regularizer=regularizers.l2(0.01),
                  # name='fc1')(x)
        # x = Dropout(0.5, name='drop_fc1')(x)
        # x = Dense(2048, activation='relu',
                  # kernel_regularizer=regularizers.l2(0.01),
                  # name='fc2')(x)
        # x = Dropout(0.5, name='drop_fc2')(x)

        # x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        # model = Model(inputs=vgg16_places.input, outputs=x)
        return model
