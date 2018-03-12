import numpy as np
import tensorflow as tf

def group_conv(x, filter_height, filter_width, num_filters, name, weights_dict, padding='SAME'):
     
  input_groups = tf.split(axis=3, num_or_size_splits=2, value=x)
  
  
  weights = np.split(weights_dict[name][0], 2, 3);
  
  bias = np.split(weights_dict[name][1], 2);
  
        
  with tf.variable_scope(name) as scope:
    conv0 = tf.layers.conv2d(input_groups[0],
                            kernel_size = [filter_height, filter_width], 
                            filters = num_filters, padding=padding,
                            kernel_initializer = tf.constant_initializer(weights[0]),
                            bias_initializer = tf.constant_initializer(bias[0]))
    print(conv0.shape)
    conv1 = tf.layers.conv2d(input_groups[1],
                            kernel_size = [filter_height, filter_width], 
                            filters = num_filters, padding=padding,
                            kernel_initializer = tf.constant_initializer(weights[1]),
                            bias_initializer = tf.constant_initializer(bias[1]))
                            
  conv = tf.concat(axis=3, values=[conv0, conv1])

  relu = tf.nn.relu(conv, name=scope.name)

  return relu


def AlexNetwork(x, weights_dict, is_training, KEEP_PROB, NUM_CLASSES):
  # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
  conv1 = tf.layers.conv2d(
    inputs=x,
    filters=96,
    kernel_size=[11, 11],
    padding="VALID",
    strides=[4,4],
    activation=tf.nn.relu,
    kernel_initializer = tf.constant_initializer(weights_dict["conv1"][0]),
    bias_initializer = tf.constant_initializer(weights_dict["conv1"][1]),
    name="conv1" )
    
 
  norm1 = tf.nn.local_response_normalization(
    conv1, depth_radius=2,
    alpha=1e-5, beta=0.75,
    name="norm1")
    
  pool1 = tf.layers.max_pooling2d(norm1, [3, 3], [2, 2], padding='VALID', name="pool1")
  
  # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
  print(pool1.shape)
  conv2 = group_conv(pool1, 5, 5, 256, 'conv2',  weights_dict)
  
  norm2 = tf.nn.local_response_normalization(
    conv2, depth_radius=2,
    alpha=1e-5, beta=0.75,
    name="norm2")

  pool2 = tf.layers.max_pooling2d(norm2, [3, 3], 2, padding='VALID', name="pool2")  
  
   # 3nd Layer: Conv (w ReLu) 
  conv3 = tf.layers.conv2d(pool2,filters = 384, kernel_size = [3, 3],
    activation=tf.nn.relu,
    kernel_initializer = tf.constant_initializer(weights_dict["conv3"][0]),
    bias_initializer = tf.constant_initializer(weights_dict["conv3"][1]),
    name="conv3" )
  
   # 4nd Layer: Conv (w ReLu) 2 groups
  conv4 = group_conv(conv3, 5, 5, 384, 'conv4', weights_dict)
  
  # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
  conv5 = group_conv(conv3, 5, 5, 256, 'conv5', weights_dict)
  pool5 = tf.layers.max_pooling2d(norm2, [3, 3], 2, padding='VALID', name="pool5")  

  # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
  flattened = tf.layers.flatten(pool5)
  fc6 = tf.layers.dense(flattened, 4096, activation=tf.nn.relu,
    kernel_initializer = tf.constant_initializer(weights_dict["fc6"][0]),
    bias_initializer = tf.constant_initializer(weights_dict["fc6"][1]),
    name='fc6')
  dropout6 = tf.layers.dropout(fc6, KEEP_PROB, training=is_training)

  # 7th Layer: FC (w ReLu) -> Dropout
  fc7 = tf.layers.dense(dropout6, 4096, activation=tf.nn.relu, 
    kernel_initializer = tf.constant_initializer(weights_dict["fc7"][0]),
    bias_initializer = tf.constant_initializer(weights_dict["fc7"][1]),
    name='fc7')
    
  dropout7 = tf.layers.dropout(fc7, KEEP_PROB, training=is_training)

  # 8th Layer: FC and return unscaled activations
  fc8 = tf.layers.dense(dropout7, NUM_CLASSES, 
    kernel_initializer = tf.constant_initializer(weights_dict["fc8"][0]),
    bias_initializer = tf.constant_initializer(weights_dict["fc8"][1]),
    name='fc8')
  return fc8
