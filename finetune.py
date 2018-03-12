from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import sklearn.model_selection as sk

import math

from PIL import Image

from os import listdir
from os.path import isfile, join
from ilio import read

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from alexnet import AlexNetwork


tfm_dir = "/mnt/Data/DeepDataset/transforms/"
BATCH_SIZE = 32
NUM_EPOCH = 20
LEARNING_RATE = 0.01
WEIGHTS_PATH = "bvlc_alexnet.npy"
# Network params
dropout_rate = 0.5
num_classes = 4
train_layers = ['fc8', 'fc7', 'fc6', 'conv5']
KEEP_PROB = 0.5
NUM_CLASSES = 1000

def get_data():

  print('Loading transforms...')
  files = [f for f in listdir(tfm_dir) if (isfile(join(tfm_dir, f)) and os.path.splitext(f)[1] == ".txt") ]

  transforms = [read(join(tfm_dir, f)).split('\n')[0].split('\t')
                  for f in files ]
                 
  transforms = [ [float(number) for number in f] for f in transforms]
  transforms = np.asarray(transforms)

  print('... done : ')
  print ( transforms.shape )

  print('Loading images...')
  img_dir = "/mnt/Data/DeepDataset/inputs_done_small/"
  filenames = [join(img_dir, f) for f in listdir(img_dir) if (isfile(join(img_dir, f)) and os.path.splitext(f)[1] == ".jpg") ]

  images = [ np.array( Image.open(f) ) for f in filenames];

  images = np.asarray(images)

  print('... done : ')
  print ( images.shape, ", ", math.floor(images.nbytes/1024/1024), 'MB' )

  images = images[:len(transforms)]


  I_train, I_test, V_train, V_test = sk.train_test_split(images, transforms, test_size=0.33, random_state=42)
  
  return I_train, I_test, V_train, V_test



def dataset_input_train_fn():
  images = tf.constant(I_train, dtype=tf.float32)
  values = tf.constant(V_train, dtype=tf.float32)

  """An input function for training"""
  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices( (images, values) )
  
  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).batch(BATCH_SIZE).repeat(NUM_EPOCH)

  # Build the Iterator, and return the read end of the pipeline.
  #    return dataset.make_one_shot_iterator().get_next()

  iterator = dataset.make_one_shot_iterator()

  return iterator.get_next()
    
        
def cnn_model_fn(features, labels, mode):

  x = tf.reshape(features, [-1, 300, 200, 1])
  # TF placeholder for graph input and output
  #x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
  #y = tf.placeholder(tf.float32, [batch_size, num_classes])
  y = labels
  keep_prob = tf.placeholder(tf.float32)

  # Initialize model
  model = AlexNet(x, keep_prob, num_classes, train_layers)
       
       
  #### Keep Only conv par to custom input size
  net = tf.layers.flatten(model.pool5)
  net = tf.layers.dropout(net, keep_prob, training = mode == tf.estimator.ModeKeys.TRAIN)
  net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
  net = tf.layers.dropout(net, keep_prob, training = mode == tf.estimator.ModeKeys.TRAIN)
  net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
  y_ = tf.layers.dense(net, 4, activation=None)

  loss = tf.losses.mean_squared_error(y, y_)

  relative_error = tf.metrics.mean_relative_error(y, y_, y)
      
  tf.identity(relative_error[1], name='train_relative_error')
  
  tf.summary.scalar('train_relative_error',relative_error[1])
  
  
  return model, loss, relative_error
  '''
  eval_metric_ops = { "accuracy": relative_error }
  
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

  train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
      
  model.load_initial_weights(sess)
            
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
  '''
  
weights_dict = np.load(WEIGHTS_PATH, encoding='bytes').item()

x = tf.placeholder(tf.float32, [None, 227, 227, 3])

  

'''
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

I_train, I_test, V_train, V_test = get_data()





model, loss, relative_error = cnn_model_fn(x, y, tf.estimator.ModeKeys.TRAIN)
'''

is_training = tf.placeholder(tf.bool, None)

net = AlexNetwork(x, weights_dict, is_training, KEEP_PROB, NUM_CLASSES)



# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    #writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    #model.load_initial_weights(sess)
    print(net.eval())

    
    
    
    
    
    
    

'''
with tf.train.MonitoredSession() as sess:
  # Since the `QueueRunners` have been started, data is available in the
  # queue, so the `sess.run(get_batch)` call will not hang.
  while not sess.should_stop():
    print(train_input_fn.eval())
'''


'''
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
images = tf.image.decode_jpeg(value, channels=3)

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  image = images.eval()
  print(image.shape)
  Image.fromarray(np.asarray(image)).show()
  
  coord.request_stop()
  coord.join(threads)
'''
