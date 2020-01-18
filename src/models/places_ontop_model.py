from keras import initializers, regularizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model, Sequential

from src.models import base_model
from src.models.keras_VGG16_places365.vgg16_places_365 import VGG16_Places365


class PlacesOntop_Model(base_model.BaseModel):
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

        if version == 8:
            return self.get_model_v8(input_shape)

        if version == 9:
            return self.get_model_v9(input_shape)

        if version == 10:
            return self.get_model_v10(input_shape)

        if version == 11:
            return self.get_model_v11(input_shape)

        if version == 12:
            return self.get_model_v12(input_shape)

        if version == 13:
            return self.get_model_v13(input_shape)

        if version == 14:
            return self.get_model_v14(input_shape)

        if version == 15:
            return self.get_model_v15(input_shape)

        if version == 16:
            return self.get_model_v16(input_shape)

        if version == 17:
            return self.get_model_v17(input_shape)

        return self.get_model_v1(input_shape)

    def get_model_v1(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape)

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Flatten(name='flatten')(x)

        x = Dense(512, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model


    def get_model_v2(self, input_shape):
        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            # pooling='max'
        )

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Flatten(name='flatten')(x)

        x = Dense(512, activation='relu',
                  name='fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v3(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            # pooling='max'
        )

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Flatten(name='flatten')(x)

        x = Dense(512, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(512, activation='relu',
                  name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)


        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v4(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            # pooling='max'
        )

        for l in vgg16_places.layers:
            l.trainable = True

        x = vgg16_places.output

        x = Flatten(name='flatten')(x)

        x = Dense(512, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(512, activation='relu',
                  name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)


        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v5(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            # pooling='max'
        )

        for l in vgg16_places.layers:
            l.trainable = True

        x = vgg16_places.output

        x = Flatten(name='flatten')(x)

        x = Dense(4096, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(2048, activation='relu',
                  name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)


        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v6(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape)

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Flatten(name='flatten')(x)

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v7(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v8(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='avg'
        )

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v9(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v10(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            if l.name.startswith('block5'):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg16_places.output

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v11(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            l.trainable = False

        x = vgg16_places.output

        x = Dense(128, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v12(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            if l.name.startswith('block5') or l.name.startswith('block4'):
                print(f'Layer {l.name} of PlacesCNN has ben set as trainable')
                l.trainable = True
            else:
                l.trainable = False

        x = vgg16_places.output

        # TODO >> set this number of units to 128 if it goes better in model v11 execution
        x = Dense(128, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v13(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            if l.name.startswith('block5') or l.name.startswith('block4'):
                print(f'Layer {l.name} of PlacesCNN has ben set as trainable')
                l.trainable = True
            else:
                l.trainable = False

        x = vgg16_places.output

        # TODO >> set this number of units to 128 if it goes better in model v11 execution
        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v14(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            if (l.name.startswith('block5') or l.name.startswith('block4')
                or l.name.startswith('block3')):
                print(f'Layer {l.name} of PlacesCNN has ben set as trainable')
                l.trainable = True
            else:
                l.trainable = False

        x = vgg16_places.output

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v15(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            if l.name.startswith('block5'):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg16_places.output

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v16(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape,
            pooling='max'
        )

        for l in vgg16_places.layers:
            if l.name.startswith('block5'):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg16_places.output

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.25, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model

    def get_model_v17(self, input_shape):

        vgg16_places = VGG16_Places365(
            weights='places', include_top=False, input_shape=input_shape)

        for l in vgg16_places.layers:
            if l.name.startswith('block5'):
                l.trainable = True
            else:
                l.trainable = False

        x = vgg16_places.output

        x = Flatten(name='flatten')(x)

        x = Dense(256, activation='relu',
                  name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(self.n_classes, activation='softmax', name="predictions")(x)

        model = Model(inputs=vgg16_places.input, outputs=x)

        return model
