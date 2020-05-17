# ScriptName: teacher_nn.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2020-05-16 13:29
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2020-05-16 13:29
# Function:
#***************************************************************#
# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf
import sys
import csv
import time

mode = 'train'
data_set = sys.argv[1]
no = sys.argv[2]

batch = 100
x_data_file = "split/"+no+data_set+'_x_train.csv'
d_data_file = "split/"+no+data_set+'_d_train.csv'
filename_queue_x = tf.train.string_input_producer([x_data_file])
filename_queue_d = tf.train.string_input_producer([d_data_file])
x_reader = tf.TextLineReader()
d_reader = tf.TextLineReader()
_, x_record = x_reader.read(filename_queue_x)
_, d_record = d_reader.read(filename_queue_d)
x_record_defaults = [[0.0]] * 16
d_record_defaults = [[0.0]] * 1
x_data = tf.cast(tf.stack(tf.decode_csv(x_record, record_defaults=x_record_defaults)), tf.float32)
d_data = tf.cast(tf.stack(tf.decode_csv(d_record, record_defaults=d_record_defaults)), tf.float32)
x_data, d_data = tf.train.batch([x_data, d_data], batch_size=batch, capacity=100+3*batch)  # [batch, 16] [batch, 1]

def linear(inputs, shape):
  w = tf.Variable(tf.random_normal(shape, -0.1 / shape[0], 0.1 / shape[0]), name='linear_w')
  return tf.matmul(inputs, w, name='result_w'), w

result, w = linear(x_data, [16, 1])
loss = tf.reduce_sum(tf.square(d_data - result))

optimizer = tf.train.AdagradOptimizer(0.01)
train = optimizer.minimize(loss)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  try:
    for i in range(2000):
      result = sess.run([train, loss, w])
      if i % 100 == 0:
        print result[1]
        sys.stdout.flush()
      if i % 100 == 0:
        print result[2]
        saver.save(sess, './checkpoint_d'+data_set+'/w', global_step = i)
    saver.save(sess, './checkpoint_d'+data_set+'/w', global_step = i)
  except tf.errors.OutOfRangeError:
    print 'finish'
  finally:
    coord.request_stop()
    coord.join(threads)
