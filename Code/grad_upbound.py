import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
from tensorflow.python.framework import ops
from PIL import Image
from six.moves import xrange

from cleverhans.utils import other_classes, pair_visual, grid_visual
from cleverhans.utils_tf import model_train, model_eval, batch_eval, model_loss
from cleverhans.attacks import fgsm, jsma
from cleverhans.attacks_tf import fgsm_sample

"""
Method to implement : Semi-Discrete Neural Network
(the function which has the slanted and non-slanted region,
in which the slanted region has gradient gamma, while the non-
slanted one has gradient 0. Also the value is being kept to 
be between 0 and M)

Here the gradient to be used shall be gamma for the slanted one,
and 0 for the others
"""

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epoch', 30, 'Number of iterations to train')
flags.DEFINE_integer('hidden_nodes1', 256, 'Number of nodes in the first hidden layer')
flags.DEFINE_integer('hidden_nodes2', 256, 'Number of nodes in the second hidden layer')
flags.DEFINE_integer('batch_size', 128, 'Size of the batch')
flags.DEFINE_float('learning_rate', 0.0001, 'Rate of learning for the optimizer')
# flags.DEFINE_float('dropout_rate', 0.2, 'Rate of dropout')
flags.DEFINE_boolean('l2Regularizer', True, 'whether we apply l2Regularize or not')
flags.DEFINE_float('reg_param', 0.1, "parameter to measure the power of regularizer")
flags.DEFINE_integer('act_grad', 2, 'the gradient of the slanted region')
flags.DEFINE_integer('upper_bound', 10, 'the upper bound of the value of activation fct')

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

act_grad = FLAGS.act_grad
reg_param = FLAGS.reg_param

#defining the activation function that should be used here
def scalar_semi_step(x_input):
    return np.clip(np.ceil(x_input), 0, upper_bound)

semi_step = np.vectorize(scalar_semi_step)

#defining its derivative
def derivative_semistep(x_input):
	x_frac = x_input % 1
	start_constant = 1.0 / act_grad

	if (x_input < 0 or x_input >= upper_bound):
		return 0
	elif x_frac <= start_constant:
		return act_grad
	else:
		return 0

d_semistep = np.vectorize(derivative_semistep)
d_semistep32 = lambda x: d_semistep(x).astype(np.float32)

def tf_d_semistep(x_input, name=None):
	with ops.name_scope(name, "d_semistep", [x_input]) as name:
		y = tf.py_func(d_semistep32, [x_input], [tf.float32], name=name, stateful=False)
		return y[0]

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

	rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e+8))
	tf.RegisterGradient(rnd_name)(grad)
	g = tf.get_default_graph()

	with g.gradient_override_map({"PyFunc" : rnd_name}):
		return tf.py_func(func, inp, Tout, stateful = stateful, name=name)

def semistep_grad(op, grad):
	x_input = op.inputs[0]

	n_grad = tf_d_semistep(x_input)
	return grad * n_grad

semi_step32 = lambda x: semi_step(x).astype(np.float32)

def tf_semistep(x_input, name=None):
	with ops.name_scope(name, "semi_step", [x_input]) as name:
		y = py_func(semi_step32, [x_input], [tf.float32], name=name, grad=semistep_grad)
		return y[0]

def semi_network(x, w1, w2, b1, b2):
	layer_1 = tf.add(tf.matmul(x, w1), b1)
	layer_1 = tf_semistep(layer_1)

	layer_out = tf.add(tf.matmul(layer_1, w2), b2)
	layer_out = tf_semistep(layer_out)

	return tf.nn.softmax(layer_out)

#storing weights and biases properly
w1 = tf.get_variable("w1", shape = [n_input, FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
w_out = tf.get_variable("w_out", shape = [FLAGS.hidden_nodes2, n_classes], initializer = tf.contrib.layers.xavier_initializer())

b1 = tf.get_variable("b1", shape = [FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable("b_out", shape = [n_classes], initializer = tf.contrib.layers.xavier_initializer())

#constructing the model
prediction = semi_network(x, w1, w_out, b1, b_out)

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

# grad_list = [2, 4, 8, 10, 16]
grad_list = [4]
upperbound_list = [15]
eps_list = [0.1]
# upperbound_list = [15, 20, 25, 50]
# upperbound_list = [20, 25]

#Initializing the variables
display_step = 1
training_epochs = FLAGS.training_epoch
count = 1
# print(img_list1)

#launching the graph
# Launch the graph
epoch_list = []
loss_list = []
reg_param = 0.001

for upper in upperbound_list:
    for grad_act in grad_list:
        for epsilon in eps_list:
            #begin running the session of the graph
            with tf.Session() as sess:
                #initializing all the global variables of the graph, which is all the parameters described
                #and also we define the value for each of the hyperparameter here
                init = tf.global_variables_initializer()
                sess.run(init)
                batch_size = FLAGS.batch_size
                act_grad = grad_act
                upper_bound = upper
                source_samples = 60

                print("currently checking for the upper bound " + str(upper_bound) + " and gradient " + str(act_grad))

                #Begin the training phase, by using only the legitimate training samples
                for epoch in range(training_epochs):
                    epoch_list.append(epoch + 1)
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
                    loss_list.append(avg_cost)

                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        print("Epoch:", '%04d' % (epoch+1), "cost=", \
                            "{:.9f}".format(avg_cost))
                print("Optimization Finished!")
                # print("Training Accuracy : " + str(100 * sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})) + " percent")
                print("Testing Accuracy : " + str(100 * sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})) + " percent")
                #End of the training with the legitimate training samples

                #firstly testing towards the whole adversarial images
                adv_images = fgsm(x, prediction, eps = epsilon)
                eval_params = {'batch_size': FLAGS.batch_size}
                X_test_adv, = batch_eval(sess, [x], [adv_images], [mnist.test.images], args=eval_params)
                assert X_test_adv.shape[0] == 10000, X_test_adv.shape

                # Evaluate the accuracy of the MNIST model on adversarial examples
                accuracy_adv = model_eval(sess, x, y, prediction, X_test_adv, mnist.test.labels,
                                      args=eval_params)
                print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

                #This time, we are trying to create the adversarial samples by using the FGSM approach
                print("\n\n")
                # print("Creating " + str(source_samples) + " * " + str(n_classes - 1) + " adversarial examples")
                # results = np.zeros((n_classes, source_samples), dtype = 'i') #to indicate whether adversarial is found for each sample and class
                # perturbs = np.zeros((n_classes, source_samples), dtype = 'f') #array of fraction of perturbed features for each sample and class

                #preparing the container for the grid visualization
                # grid_shape = (n_classes, n_classes, 28, 28, 1) #still being specific for MNIST dataset
                # grid_vis_data = np.zeros(grid_shape, dtype = 'f')

                print("---------------------------------------------------------")
                print("Creating adversarial image for " + str(source_samples) + " examples")
                #looping over the samples that is about to be perturbed to form the adversarial samples
                for sample_index in xrange(0, source_samples):
                    current_class = int(np.argmax(mnist.test.labels[sample_index]))  
                    target_class = other_classes(n_classes, current_class)

                    #keeping the original images along the diagonal for the grid visualization
                    # grid_vis_data[current_class, current_class, :, :, :] = np.reshape(
                        # mnist.test.images[sample_index:(sample_index + 1)], (28, 28, 1))

                    #now loop over all the target classes
                    # for target in target_class:


                    #now running the FGSM approach
                    # print(x.get_shape())
                    adv_x = fgsm_sample(sess, x, prediction, eps = epsilon, sample=mnist.test.images[sample_index : (sample_index + 1)])
                    pred_adv = sess.run(prediction, feed_dict = {x : adv_x})
                    pred_real = sess.run(prediction, feed_dict = {x : mnist.test.images[sample_index : (sample_index + 1)]})
                    class_adv = int(np.argmax(pred_adv))
                    class_real = int(np.argmax(pred_real))
                    # print(adv_x.get_shape())

                    #displaying the original and adversarial images side-by-side
                    if 'figure' not in vars():
                        if class_adv == class_real:
                            figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' robust 020517-4 disc2 ' + str(sample_index) + '.jpg')
                        else:
                            figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' fail 020517-4 disc2 ' + str(sample_index) + ' pred ' + str(class_real) + " adv_label " + str(class_adv) + '.jpg')
                    else:
                        if class_adv == class_real:
                            figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' robust 020517-4 disc2 ' + str(sample_index) + '.jpg', figure=figure)
                        else:
                            figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' fail 020517-4 disc2 ' + str(sample_index) + ' pred ' + str(class_real) + " adv_label " + str(class_adv) + '.jpg', figure=figure)
                    # grid_vis_data[target, current_class, :, :, :] = np.reshape(adv_x, (28, 28, 1))

            # _ = grid_visual(data = grid_vis_data, name = '020517-1 grid sample.jpg')

                             

                # adv_x = fgsm(x, prediction, eps = 0.3)
                # X_test_adv, = batch_eval(sess, [x], [adv_x], [mnist.test.images], args={'batch_size' : 128})

                # accuracy_adv = model_eval(sess, x, y, prediction, X_test_adv, mnist.test.labels,
                #                       args={'batch_size' : 128})
                # print('Test accuracy on adversarial examples: ' + str(accuracy_adv * 100) + ' percent')
            # f.close()