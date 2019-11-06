from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf


shape = [2, 2, 2]
n_cell_dim = 2

def init_vars(sess):
  sess.run(tf.global_variables_initializer())


def train_graph():
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    with tf.Session() as sess:
      is_training = True

      inputs = tf.random_uniform(shape, dtype=tf.float32)

      lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
          num_layers=1,
          num_units=n_cell_dim,
          direction='bidirectional',
          dtype=tf.float32)
      lstm.build(inputs.get_shape())
      outputs, output_states = lstm(inputs, training=is_training)

      with tf.device('/cpu:0'):
        saver = tf.train.Saver()

      init_vars(sess)
      saver.save(sess, '/tmp/model')


def inf_graph():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.Session() as sess:
      single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
          n_cell_dim, reuse=tf.get_variable_scope().reuse)

      inputs = tf.random_uniform(shape, dtype=tf.float32)
      lstm_fw_cell = [single_cell() for _ in range(1)]
      lstm_bw_cell = [single_cell() for _ in range(1)]
      (outputs, output_state_fw,
       output_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
           lstm_fw_cell,
           lstm_bw_cell,
           inputs,
           dtype=tf.float32,
           time_major=True)
      saver = tf.train.Saver()

      saver.restore(sess, '/tmp/model')
      print(sess.run(outputs))


def main(unused_argv):
  train_graph()
  inf_graph()


if __name__ == '__main__':
  tf.app.run(main)
