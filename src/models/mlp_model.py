from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from src.models import base_model

class MLP_Model(base_model.BaseModel):
    def __init__(self, batch_size, n_classes, epochs, img_size, n_channels,):
        super().__init__(batch_size, n_classes, epochs, img_size, n_channels,
                         model_name=type(self).__name__)

        input_shape = (img_size, img_size, n_channels)
        self.model = self.get_model(input_shape=input_shape)

    def get_model(self, input_shape, ):

        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model
