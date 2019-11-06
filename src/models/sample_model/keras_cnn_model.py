import tensorflow as tf
import optimizers as opt
import losses as loss
import metrics

class KerasCnnModel(tf.keras.Model):

    def __init__(self,):
        super(KerasCnnModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(6, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)

        if training:
            x = self.dropout(x, training=training)

        return self.dense2(x)

def main():

    model = KerasCnnModel()

    model.compile(optimizer=opt.get_adam_optimizer(),
                  loss=loss.get_softmax_cross_entropy(),
                  metrics=metrics.get_categorical_cross_entropy()
                  )
    return model
