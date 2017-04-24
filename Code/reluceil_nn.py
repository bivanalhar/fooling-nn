import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Method being Implemented : ReLU-Ceil Neural Network
(having the ReLU activation function for the training phase,
while we are using the Ceiling activation function for the
testing phase)
"""

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epoch', 150, 'Number of iterations to train')
flags.DEFINE_integer('hidden_nodes1', 256, 'Number of nodes in the first hidden layer')
flags.DEFINE_integer('hidden_nodes2', 256, 'Number of nodes in the second hidden layer')
flags.DEFINE_integer('batch_size', 128, 'Size of the batch')
flags.DEFINE_float('learning_rate', 0.0001, 'Rate of learning for the optimizer')
# flags.DEFINE_float('dropout_rate', 0.2, 'Rate of dropout')
flags.DEFINE_boolean('l2Regularizer', True, 'whether we apply l2Regularize or not')
flags.DEFINE_float('reg_param', 0.1, "parameter to measure the power of regularizer")

n_input = 784 #for now, keep on focusing for the MNIST data
n_hidden = 128 #defining the number of nodes in the hidden layer
n_classes = 10 #remember that MNIST has 10 classes

#importing the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

l2_regularize = True

#defining the graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#defining the first neural network model to be used
def relu_network(input_, w1_shape, w2_shape, b1_shape, b2_shape):
	with tf.variable_scope("relu_ceil") as scope:
		weight1 = tf.get_variable("weight1", w1_shape, initializer = tf.random_normal_initializer())
		bias1 = tf.get_variable("bias1", b1_shape, initializer = tf.constant_initializer(0.1))

		weight2 = tf.get_variable("weight2", w2_shape, initializer = tf.random_normal_initializer())
		bias2 = tf.get_variable("bias2", b2_shape, initializer = tf.constant_initializer(0.1))

		layer_1 = tf.add(tf.matmul(input_, weight1), bias1)
		layer_1 = tf.nn.relu(layer_1)

		layer_2 = tf.add(tf.matmul(layer_1, weight2), bias2)
		layer_2 = tf.nn.relu(layer_2)

		return tf.nn.softmax(layer_2), weight1, bias1, weight2, bias2

#defining the second neural network model to be used
def ceil_network(input_, w1_shape, w2_shape, b1_shape, b2_shape):
	with tf.variable_scope("relu_ceil") as scope:
		scope.reuse_variables()

		weight1 = tf.get_variable("weight1", w1_shape, initializer = tf.random_normal_initializer())
		bias1 = tf.get_variable("bias1", b1_shape, initializer = tf.constant_initializer(0.1))

		weight2 = tf.get_variable("weight2", w2_shape, initializer = tf.random_normal_initializer())
		bias2 = tf.get_variable("bias2", b2_shape, initializer = tf.constant_initializer(0.1))

		layer_1 = tf.add(tf.matmul(input_, weight1), bias1)
		layer_1 = tf.ceil(tf.nn.relu(layer_1))

		layer_2 = tf.add(tf.matmul(layer_1, weight2), bias2)
		layer_2 = tf.ceil(tf.nn.relu(layer_2))

		return tf.nn.softmax(layer_2)

prediction_1, weight1, bias1, weight2, bias2 = relu_network(x, [n_input, n_hidden], [n_hidden, n_classes], [n_hidden], [n_classes])
prediction_2 = ceil_network(x, [n_input, n_hidden], [n_hidden, n_classes], [n_hidden], [n_classes])

cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction_1,1e-10,1.0)))

if l2_regularize:
    cost = tf.reduce_mean(cross_entropy) + tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2) + tf.nn.l2_loss(bias1) + tf.nn.l2_loss(bias2)
else:
    cost = tf.reduce_mean(cross_entropy)
# cost = tf.Print(cost, [cost], "cost = ", summarize = 30)
optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(cost)

#Initializing the variables
init = tf.global_variables_initializer()
display_step = 1
training_epochs = FLAGS.training_epoch
# epoch_list = []
# loss_list = []

#launching the graph
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    batch_size = FLAGS.batch_size

    # Training cycle
    for epoch in range(training_epochs):
        # epoch_list.append(epoch + 1)
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # total_batch = 2
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # loss_list.append(avg_cost)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # plt.plot(epoch_list, loss_list)
    # plt.xlabel("Epoch")
    # plt.ylabel("Cost Function")

    # plt.title("Discretized Activation NN")

    # plt.savefig("Discrete_NN_function.png")

    # plt.clf()

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction_2, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Training Accuracy:", 100 * sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}), "percent")
    print("Testing Accuracy:", 100 * sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}), "percent")