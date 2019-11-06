import tensorflow as tf


# TODO>> SEARCH HOW TO USE WEIGHTED LOSS

def get_softmax_cross_entropy():
    return tf.losses.softmax_cross_entropy

def get_sparse_softmax_cross_entropy():
    return tf.losses.sparse_softmax_cross_entropy

