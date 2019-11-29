import tensorflow as tf
import tf_metrics
from tensorflow import keras

class CategoricalTruePositives(keras.metrics.Metric):

    def __init__(self, name='categorical_true_positives', **kwargs):
      super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.argmax(y_pred)
      values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
      values = tf.cast(values, 'float32')
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))

    '''
    def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
      values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
      values = tf.cast(values, 'float32')
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))
    '''
    def result(self):
      return self.true_positives

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.true_positives.assign(0.)


def multi_class_recall(num_classes, average, weights, **kwargs):
    def categorical_recall(labels, predictions):

        label_weights = [weights[idx] for idx, w in enumerate(weights)]

        # any tensorflow metric
        value, update_op = tf_metrics.recall(
            labels, predictions, num_classes,
            average=average, weights=tf.constant(label_weights), **kwargs)

        # find all variables created for this metric
        #print([len(i.name.split('/')) for i in tf.local_variables()])
        metric_vars = []
        for i in tf.local_variables():
            if len(i.name.split('/')) > 2 and 'categorical_recall' in i.name.split('/')[1]:
                metric_vars.append(i)

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value
    return categorical_recall
