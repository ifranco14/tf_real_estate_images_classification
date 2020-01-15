from keras import initializers, regularizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.applications import VGG16

from src.models import base_model

class ImageNetOntop_Model(base_model.BaseModel):
    def __init__(self, batch_size, n_classes, epochs, img_size, n_channels,
                 version=1):
        if img_size != 224:
            img_size = 224
            print(f'Warning!! The images will be resized to 224 instead of {img_size}',
                  'because the pretrained model uses that shape')
        super().__init__(batch_size, n_classes, epochs, img_size, n_channels,
                         model_name=type(self).__name__)

        input_shape = (img_size, img_size, n_channels)
        self.model = self.get_model(input_shape=input_shape, version=version)

    def get_model(self, input_shape, version=1):

        if version == 1:
            return self.get_model_v1(input_shape)

        if version == 2:
            return self.get_model_v2(input_shape)

        if version == 3:
            return self.get_model_v3(input_shape)

        if version == 4:
            return self.get_model_v4(input_shape)

        if version == 5:
            return self.get_model_v5(input_shape)

        if version == 6:
            return self.get_model_v6(input_shape)

        if version == 7:
            return self.get_model_v7(input_shape)

        # if version == 8:
            # return self.get_model_v8(input_shape)

        return self.get_model_v1(input_shape)

    def get_model_v1(self, input_shape):

        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,)

        for l in vgg_model.layers:
            l.trainable = False

        x = vgg_model.output

        x = Flatten(name='flatten')(x)

        x = Dense(512, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg_model.input, outputs=x)

        return model

    def get_model_v2(self, input_shape):

        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,)

        for l in vgg_model.layers:
            l.trainable = False

        x = vgg_model.output

        x = Flatten(name='flatten')(x)

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg_model.input, outputs=x)

        return model

    def get_model_v3(self, input_shape):

        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,
                          pooling='max')

        for l in vgg_model.layers:
            l.trainable = False

        x = vgg_model.output

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg_model.input, outputs=x)

        return model

    def get_model_v4(self, input_shape):

        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,)

        for l in vgg_model.layers:
            if l.name.startswith('block5'):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg_model.output

        x = Flatten(name='flatten')(x)

        x = Dense(512, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg_model.input, outputs=x)

        return model

    def get_model_v5(self, input_shape):

        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,)

        for l in vgg_model.layers:
            if l.name.startswith('block5'):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg_model.output

        x = Flatten(name='flatten')(x)

        x = Dense(1024, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg_model.input, outputs=x)

        return model

    def get_model_v6(self, input_shape):

        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,)

        for l in vgg_model.layers:
            if (l.name.startswith('block5')
                or l.name.startswith('block4')):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg_model.output

        x = Flatten(name='flatten')(x)

        x = Dense(1024, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg_model.input, outputs=x)

        return model


    def get_model_v7(self, input_shape):

        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape,)

        for l in vgg_model.layers:
            if l.name.startswith('block5'):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg_model.output

        x = Flatten(name='flatten')(x)

        x = Dense(2048, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg_model.input, outputs=x)

        return model
