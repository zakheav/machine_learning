# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf
import sys
import csv

data_set = 'real'
if len(sys.argv) > 1:
  data_set = sys.argv[1]

batch = 1
x_data_file = data_set+'_x_test.csv'
d_data_file = data_set+'_d_test.csv'
w_data_file = data_set+'_w_test.csv'
filename_queue_x = tf.train.string_input_producer([x_data_file])
filename_queue_d = tf.train.string_input_producer([d_data_file])
filename_queue_w = tf.train.string_input_producer([w_data_file])
x_reader = tf.TextLineReader()
d_reader = tf.TextLineReader()
w_reader = tf.TextLineReader()
_, x_record = x_reader.read(filename_queue_x)
_, d_record = d_reader.read(filename_queue_d)
_, w_record = w_reader.read(filename_queue_w)
x_record_defaults = [[0.0]] * 16
d_record_defaults = [[0.0]] * 1
w_record_defaults = [[0.0]] * 16
x_data = tf.cast(tf.stack(tf.decode_csv(x_record, record_defaults=x_record_defaults)), tf.float32)
d_data = tf.cast(tf.stack(tf.decode_csv(d_record, record_defaults=d_record_defaults)), tf.float32)
w_data = tf.cast(tf.stack(tf.decode_csv(w_record, record_defaults=w_record_defaults)), tf.float32)
x_data, w_data, d_data = tf.train.batch([x_data, w_data, d_data], batch_size=batch, capacity=100+3*batch)  # [batch, 16] [batch, 16] [batch, 1]

def gat(inputs, dim, feature_trans, self_weight):
  inputs = tf.reshape(inputs, [-1, 16, 1])
  tile_inputs = tf.tile(inputs, [1, 1, dim])  # [batch, 16, dim]
  trans_inputs = tile_inputs * tf.reshape(feature_trans, [1, 16, dim])  # [batch, 16, dim]
  att_weight = tf.nn.softmax(tf.matmul(trans_inputs, tf.transpose(trans_inputs, [0, 2, 1])))  # [batch, 16, 16]

  att_output = tf.matmul(att_weight, trans_inputs) + tf.reshape(self_weight, [1, 16, dim]) * trans_inputs  # [batch, 16, dim]
  att_fc = tf.Variable(tf.random_normal([dim, 1], -0.1, 0.1), name='att_fc')  # [dim, 1]
  output = tf.reshape(tf.matmul(tf.reshape(att_output, [-1, dim]), att_fc), [-1, 16])  # [batch, 16]
  return output

def fc(inputs, w, b):
  return tf.nn.leaky_relu(tf.matmul(inputs, w) + b)

def linear(inputs, w):
  return tf.matmul(inputs, w)

with tf.Session() as sess:
  saver = tf.train.import_meta_graph('./checkpoint_ts'+data_set+'/w-499.meta')
  saver.restore(sess, './checkpoint_ts'+data_set+'/w-499')
  graph = tf.get_default_graph()
  fea_trans = graph.get_tensor_by_name('fea_trans:0')
  self_weight = graph.get_tensor_by_name('self_weight:0')
  fc_w0 = graph.get_tensor_by_name('fc_w_6:0')
  fc_b0 = graph.get_tensor_by_name('fc_b_6:0')
  fc_w1 = graph.get_tensor_by_name('fc_w_7:0')
  fc_b1 = graph.get_tensor_by_name('fc_b_7:0')
  fc_w2 = graph.get_tensor_by_name('fc_w_8:0')
  fc_b2 = graph.get_tensor_by_name('fc_b_8:0')

  linear_w0 = graph.get_tensor_by_name('linear_w_1:0')

  x_data = gat(x_data, fea_trans, self_weight)
  s_h1 = fc(x_data, fc_w0, fc_b0)
  s_h2 = fc(s_h1, fc_w1, fc_b1)
  s_h3 = fc(s_h2, fc_w2, fc_b2)
  s_result_w = linear(s_h3, linear_w0)

  loss1 = tf.reduce_sum(tf.square(s_result_w - w_data))

  result_w = tf.reshape(tf.reduce_mean(s_result_w, [0]), [1, 16])
  d = tf.matmul(x_data, tf.transpose(result_w))
  loss2 = tf.reduce_sum(tf.square(d_data - d))

  # run test
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  try:
    for i in range(100):
      result = sess.run([loss1, loss2])
      print result[0], result[1]
  except tf.errors.OutOfRangeError:
    print 'finish'
  finally:
    coord.request_stop()
    coord.join(threads)
