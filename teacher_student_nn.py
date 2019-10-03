# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf
import sys
import csv

data_set = 'real'
if len(sys.argv) > 1:
  data_set = sys.argv[1]

batch = 6000
x_data_file = data_set+'_x_train.csv'
w_data_file = data_set+'_w_train.csv'
filename_queue_x = tf.train.string_input_producer([x_data_file])
filename_queue_w = tf.train.string_input_producer([w_data_file])
x_reader = tf.TextLineReader()
w_reader = tf.TextLineReader()
_, x_record = x_reader.read(filename_queue_x)
_, w_record = w_reader.read(filename_queue_w)
x_record_defaults = [[0.0]] * 16
w_record_defaults = [[0.0]] * 16
x_data = tf.cast(tf.stack(tf.decode_csv(x_record, record_defaults=x_record_defaults)), tf.float32)
w_data = tf.cast(tf.stack(tf.decode_csv(w_record, record_defaults=w_record_defaults)), tf.float32)
x_data, w_data = tf.train.batch([x_data, w_data], batch_size=batch, capacity=100+3*batch)  # [batch, 16]

def fc(inputs, shape, w=None, b=None, mode='train'):
  if w == None and b == None:
    w = tf.Variable(tf.random_normal(shape, -0.1 / shape[0], 0.1 / shape[0]), name='fc_w')
    b = tf.Variable(tf.random_normal([shape[1]], -1, 1), name='fc_b')
  r = tf.nn.leaky_relu(tf.matmul(inputs, w) + b)
  if mode == 'test':
    return r
  else:
    return tf.nn.dropout(r, 0.5)

def linear(inputs, shape, w=None):
  if w == None:
    w = tf.Variable(tf.random_normal(shape, -0.1 / shape[0], 0.1 / shape[0]), name='linear_w')
  return tf.matmul(inputs, w)

def expand_data(inputs):
  return tf.random_normal(tf.shape(inputs), 0.0, 0.1) + inputs

with tf.Session() as sess:
  # teacher model
  saver = tf.train.import_meta_graph('./checkpoint_d'+data_set+'/w-499.meta')
  saver.restore(sess, './checkpoint_d'+data_set+'/w-499')
  graph = tf.get_default_graph()
  fc_w0 = graph.get_tensor_by_name('fc_w:0')
  fc_b0 = graph.get_tensor_by_name('fc_b:0')
  fc_w1 = graph.get_tensor_by_name('fc_w_1:0')
  fc_b1 = graph.get_tensor_by_name('fc_b_1:0')
  fc_w2 = graph.get_tensor_by_name('fc_w_2:0')
  fc_b2 = graph.get_tensor_by_name('fc_b_2:0')
  fc_w3 = graph.get_tensor_by_name('fc_w_3:0')
  fc_b3 = graph.get_tensor_by_name('fc_b_3:0')
  fc_w4 = graph.get_tensor_by_name('fc_w_4:0')
  fc_b4 = graph.get_tensor_by_name('fc_b_4:0')
  fc_w5 = graph.get_tensor_by_name('fc_w_5:0')
  fc_b5 = graph.get_tensor_by_name('fc_b_5:0')

  linear_w0 = graph.get_tensor_by_name('linear_w:0')

  expand_data = expand_data(x_data)
  x_expand_data = tf.concat([x_data, expand_data], 0)

  # 只拟合扩展数据
  h1 = fc(expand_data, [16, 256], fc_w0, fc_b0, 'test')
  h2 = fc(h1, [256, 256], fc_w1, fc_b1, 'test')
  h3 = fc(h2, [256, 256], fc_w2, fc_b2, 'test')
  h4 = fc(h3, [256, 128], fc_w3, fc_b3, 'test')
  h5 = fc(h4, [128, 128], fc_w4, fc_b4, 'test')
  h6 = fc(h5, [128, 64], fc_w5, fc_b5, 'test')
  t_result_w = tf.stop_gradient(linear(h6, [64, 16], linear_w0))

  w_expand_data = tf.concat([w_data, t_result_w], 0)

  # student model
  s_h1 = fc(x_expand_data, [16, 256], None, None, 'test')
  s_h2 = fc(s_h1, [256, 256], None, None, 'test')
  s_h3 = fc(s_h2, [256, 128], None, None, 'test')
  s_result_w = linear(s_h3, [128, 16], None)

  loss = tf.reduce_sum(tf.square(s_result_w - w_expand_data))
  optimizer = tf.train.AdamOptimizer(0.001)
  train = optimizer.minimize(loss)

  # loss1 = tf.reduce_sum(tf.square(w_data - t_result_w))
  # loss2 = tf.reduce_sum(tf.square(w_data - s_result_w))

  # init uinit variables
  uninit_vars = []
  for var in tf.all_variables():
    try:
      sess.run(var)
    except tf.errors.FailedPreconditionError:
      uninit_vars.append(var)
  sess.run(tf.initialize_variables(uninit_vars))

  # run train
  new_saver = tf.train.Saver()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  try:
    for i in range(500):
      result = sess.run([train, loss])
      print result[1]  #, result[2], result[3]
      sys.stdout.flush()
    new_saver.save(sess, './checkpoint_ts'+data_set+'/w', global_step = i)
  except tf.errors.OutOfRangeError:
    print 'finish'
  finally:
    coord.request_stop()
    coord.join(threads)
