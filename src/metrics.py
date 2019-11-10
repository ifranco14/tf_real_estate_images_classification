import tensorflow as tf


# TODO>> SEARCH HOW TO USE sample_weight PARAM of update_state() internal method
def get_categorical_cross_entropy():
    """
    When using a one_hot representation.
    When labels values are [2, 0, 1], y_true = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    """
    return tf.keras.metrics.CategoricalCrossentropy()

def get_sparse_categorical_cross_entropy():
    """
    When using a scalar representation.
    y_true = [1, 2, 1, 3]
    """
    return tf.keras.metrics.SparseCategoricalCrossentropy()
