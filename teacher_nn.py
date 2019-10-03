# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf
import sys
import csv
import time

mode = 'train'
data_set = 'real'
if len(sys.argv) > 1:
  mode = sys.argv[1]
  data_set = sys.argv[2]

if mode == 'train':
  batch = 6000
  x_data_file = data_set+'_x_train.csv'
  d_data_file = data_set+'_d_train.csv'
  w_data_file = data_set+'_w_train.csv'
else:
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

def fc(inputs, shape, mode):
  w = tf.Variable(tf.random_normal(shape, -0.1 / shape[0], 0.1 / shape[0]), name='fc_w')
  b = tf.Variable(tf.random_normal([shape[1]], -1, 1), name='fc_b')
  r = tf.nn.leaky_relu(tf.matmul(inputs, w) + b)
  if mode == 'test':
    return r
  else:
    return tf.nn.dropout(r, 0.5)

def linear(inputs, shape):
  w = tf.Variable(tf.random_normal(shape, -0.1 / shape[0], 0.1 / shape[0]), name='linear_w')
  return tf.matmul(inputs, w, name='result_w'), w

h1 = fc(x_data, [16, 256], mode)
h2 = fc(h1, [256, 256], mode)
h3 = fc(h2, [256, 256], mode)
h4 = fc(h3, [256, 128], mode)
h5 = fc(h4, [128, 128], mode)
h6 = fc(h5, [128, 64], mode)
result_w, _ = linear(h6, [64, 16])
loss1 = tf.reduce_sum(tf.square(w_data - result_w))

result_w = tf.reshape(tf.reduce_mean(result_w, [0]), [1, 16])
d = tf.matmul(x_data, tf.transpose(result_w))
loss2 = tf.reduce_sum(tf.square(d_data - d))

loss = loss1
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

if mode == 'train':
  saver = tf.train.Saver()
  init = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      for i in range(500):
        result = sess.run([train, loss, loss1, loss2])
        if i % 1 == 0:
          print result[1]
          sys.stdout.flush()
        if i % 100 == 0:
          saver.save(sess, './checkpoint_d'+data_set+'/w', global_step = i)
      saver.save(sess, './checkpoint_d'+data_set+'/w', global_step = i)
    except tf.errors.OutOfRangeError:
      print 'finish'
    finally:
      coord.request_stop()
      coord.join(threads)
else:
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, './checkpoint_d'+data_set+'/w-499')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      for i in range(100):
        result = sess.run([loss1, loss2])
        print result[1]
    except tf.errors.OutOfRangeError:
      print 'finish'
    finally:
      coord.request_stop()
      coord.join(threads)
