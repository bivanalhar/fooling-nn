# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
from tensorflow.python.framework import ops
from PIL import Image

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

lin_sigmoid = lambda x: 0.25 * x + 0.5
HardSigmoid = lambda x: tf.minimum(tf.maximum(lin_sigmoid(x), 0.), 1.)

# def NHardSigmoid(x,
#                  use_noise,
#                  c=0.05):
#     """
#     Noisy Hard Sigmoid Units: NANI as proposed in the paper
#     ----------------------------------------------------
#     Arguments:
#         x: theano tensor variable, input of the function.
#         use_noise: bool, whether to add noise or not to the activations, this is in particular
#         useful for the test time, in order to disable the noise injection.
#         c: float, standard deviation of the noise
#     """

#     def noise_func() :return tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
#     def zero_func (): return tf.zeros(tf.shape(x), dtype=tf.float32, name=None)

#     noise=tf.cond(use_noise,noise_func,zero_func)

#     res = HardSigmoid(x + c * noise)
#     return res

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return tf.nn.softmax(tf.nn.relu(out_layer))

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

img_input1 = Image.open("resized_original_7.png")
img_input2 = Image.open("resized_adversarial_7.png")
img_list1 = np.divide(np.reshape(np.asarray(img_input1).ravel(), (-1, n_input)), 255)
img_list2 = np.divide(np.reshape(np.asarray(img_input2).ravel(), (-1, n_input)), 255)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    print("Prediction vector original: " + str(sess.run(pred, feed_dict={x : img_list1})))
    print("Prediction vector adversarial: " + str(sess.run(pred, feed_dict={x : img_list2})))

    # with tf.device("/gpu:0"):
       #  # Test model
       #  correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
       #  # Calculate accuracy
       #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # f.write("Training Accuracy:" + str(100 * sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})) + "percent\n")
    # f.write("Testing Accuracy:" + str(100 * sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})) + "percent\n\n")

    # count += 1
