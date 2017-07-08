import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

#defining the graph input
#First step : Defining its placeholder
x = tf.placeholder("float", shape = [None, 784])
y = tf.placeholder("float", shape = [None, 10])

def bang_network(x):
	#just use the usual vanilla MLP
	layer_1 = tf.add(tf.matmul(x, w1), b1)
	layer_1 = tf.nn.relu(layer_1)

	layer_out = tf.add(tf.matmul(layer_1, w_out), b_out)
	layer_out = tf.nn.relu(layer_out)

	return layer_out

prediction = bang_network(x)

with tf.device("/gpu:0"):
	cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

	if l2_regularize:
	    cost = tf.reduce_mean(cross_entropy) + reg_param * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w_out) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b_out))
	else:
	    cost = tf.reduce_mean(cross_entropy)
	# cost = tf.Print(cost, [cost], "cost = ", summarize = 30)
optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(cost)

with tf.device("/gpu:0"):
    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))