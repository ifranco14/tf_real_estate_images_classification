from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from src.data import data
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode,):
    """
    Model function for CNN
    """
    # input_layer = tf.reshape(features['x'],
    #               [batch_size, img_height, img_width, channels])

    IMG_SIZE = 128
    CHANNELS = 3
    LEARNING_RATE = 0.001

    input_layer = tf.reshape(features['x'], [-1, IMG_SIZE, IMG_SIZE, CHANNELS])

    # Conv 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=int(IMG_SIZE / 4),
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        )

    # Pool 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)
    # Conv 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=int(IMG_SIZE / 2),
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
    )

    # Pool 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2)

    # FC
    pool2_flat = tf.reshape(pool2, [-1, int(IMG_SIZE / 4) \
                                    * int(IMG_SIZE / 4) \
                                    * int(IMG_SIZE / 2)])

    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)


    # Dropout
    dropout = tf.layers.dropout(inputs=dense2,
                                rate=.3,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # logits = tf.layers.dense(inputs=dropout, units=n_classes)
    logits = tf.layers.dense(inputs=dropout, units=6)

    predictions = {
        # Generate predictions
        'classes': tf.argmax(input=logits, axis=1),
        # Add softmax tensor, used for PREDICT and by the logging_hook
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metrics_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'], weights=None)
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


def main():
    # Load training and eval data

    X, y = data.load_dataset(dataset='rei',)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=True)

    X_train = X_train/np.float32(255)
    # X_train = X_train.astype(np.int32)  # not required

    X_test = X_test/np.float32(255)
    # X_test = X_test.astype(np.int32)  # not required

    # estimator instance
    clf = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                 model_dir="/tmp/clf")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # train one step and display the probabilties
    clf.train(
        input_fn=train_input_fn,
        steps=50000,)
        # hooks=[logging_hook])


    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)

    eval_results = clf.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == '__main__':
    main()
