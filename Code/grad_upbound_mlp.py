import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
import os
import time

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from PIL import Image
from six.moves import xrange

from cleverhans.utils import other_classes, pair_visual, grid_visual
from cleverhans.utils_tf import model_train, model_eval, batch_eval, model_loss, model_evalarray, model_argmax
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod, jsma
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
flags.DEFINE_integer('nb_classes', 10, 'number of classes available in the dataset')
flags.DEFINE_integer('source_samples', 60, 'number of samples to be tested')
flags.DEFINE_boolean('train_mode', True, 'determiner on whether we run the training or testing mode')

n_input = 784 #for now, keep on focusing for the MNIST data
n_hidden = 128 #defining the number of nodes in the hidden layer
n_classes = 10 #remember that MNIST has 10 classes

#importing the MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

l2_regularize = True

#defining the graph input
#now starting the main part of the execution flow
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

act_grad = FLAGS.act_grad
reg_param = FLAGS.reg_param
upper_bound = FLAGS.upper_bound

#defining the activation function that should be used here
def scalar_semi_step(x_input):
    return np.clip(np.ceil(x_input), 0, upper_bound)

semi_step = np.vectorize(scalar_semi_step)

#defining its derivative
def derivative_semistep(x_input):
	x_frac = x_input % 1
	start_constant = 1.0 / act_grad

	if (x_input < 0 or x_input > upper_bound):
		return 1e-10
	elif x_frac <= start_constant:
		return act_grad
	else:
		return 1e-10

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

def semi_network(x):
    # x_input = tf.reshape(x, (-1, 784))
    layer_1 = tf.add(tf.matmul(x, w1), b1)
    layer_1 = tf_semistep(layer_1)

    layer_out = tf.add(tf.matmul(layer_1, w_out), b_out)
    layer_out = tf_semistep(layer_out)

    return tf.nn.softmax(layer_out)

#storing weights and biases properly
w1 = tf.get_variable("w1", shape = [n_input, FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
w_out = tf.get_variable("w_out", shape = [FLAGS.hidden_nodes2, n_classes], initializer = tf.contrib.layers.xavier_initializer())

b1 = tf.get_variable("b1", shape = [FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable("b_out", shape = [n_classes], initializer = tf.contrib.layers.xavier_initializer())

#constructing the model
prediction = semi_network(x)

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

grad_list = [2, 4, 8]
upperbound_list = [15, 20]
eps_list = [0.3]
batch_size = FLAGS.batch_size

#Initializing the variables
display_step = 1
training_epochs = FLAGS.training_epoch
count = 1

current_val_acc = 0.0

# var = [v for v in tf.trainable_variables()]
# print(var)

# the network for the MNIST discrete will be saved in mnist_discrete.ckpt
# as for the MLP, it will be saved in mnist_vanilla.ckpt
saver = tf.train.Saver()
save_path = os.path.join("/tmp", "mnist_discrete.ckpt")

if FLAGS.train_mode:
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

                    # print("currently checking for the upper bound " + str(upper_bound) + " and gradient " + str(act_grad))

                    #Begin the training phase, by using only the legitimate training samples
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

                    val_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
                    print("validation accuracy is " + str(100 * val_acc) + " percent")
                    if val_acc > current_val_acc:
                        print("best validation reached at grad " + str(grad_act) + " and upper bound " + str(upper))
                        current_val_acc = val_acc
                        saver.save(sess, save_path)

else:
    act_grad = 2
    upper_bound = 20

    count_true = 0
    count_adv = 0

    for epsilon in eps_list:
        with tf.Session() as sess:
            saver.restore(sess, save_path)
            test_acc = sess.run(accuracy, feed_dict={x : mnist.test.images, y : mnist.test.labels})
            print("Test accuracy is " + str(100 * test_acc) + " percent")

            #now creating the JSMA approach (Jacobian-based Saliency Map Approach)
            result_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'i')
            perturb_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'f')

            # grid_shape = (n_classes, n_classes, 28, 28, 1)
            # grid_viz_data = np.zeros(grid_shape, dtype = 'f')

            start_time = time.time()
            jsma = SaliencyMapMethod(semi_network, back='tf', sess=sess)
            for i in xrange(0, FLAGS.source_samples):
                current_class = int(np.argmax(mnist.test.labels[i:(i+1)]))
                target_class = other_classes(FLAGS.nb_classes, current_class)

                for target in target_class:
                    one_hot_target = np.zeros((1, FLAGS.nb_classes), dtype = np.float32)
                    one_hot_target[0, target] = 1
                    jsma_params = {'theta': 1., 'gamma': 0.1,
                                'nb_classes': FLAGS.nb_classes, 'clip_min': 0.,
                                'clip_max': 1., 'targets': y,
                                'y_val': one_hot_target}
                    adv_x = jsma.generate_np(mnist.test.images[i:i+1], **jsma_params)

                    #checking for the success rate
                    res = int(model_argmax(sess, x, prediction, adv_x) == target)

                    result_overall[target, i] = res

            n_targets_tried = (n_classes - 1) * FLAGS.source_samples
            success_rate = float(np.sum(result_overall)) / n_targets_tried
            print('Execution time for adv. examples {0:.4f} seconds'.format(time.time() - start_time))
            print('Avg. rate of successful adv. examples {0:.4f}'.format(success_rate))


#         #firstly testing towards the whole adversarial images
#         fgsm = FastGradientMethod(semi_network, sess = sess)
#         fgsm_params = {'eps' : 0.3}
#         adv_images = fgsm.generate(x, **fgsm_params)
#         preds_adv = semi_network(adv_images)

#         # adv_images = fgsm(x, prediction, eps = epsilon)
#         eval_params = {'batch_size': FLAGS.batch_size}
#         # X_test_adv, = batch_eval(sess, [x], [adv_images], [mnist.test.images], args=eval_params)
#         # assert X_test_adv.shape[0] == 10000, X_test_adv.shape

#         # Evaluate the accuracy of the MNIST model on adversarial examples
#         accuracy_adv = model_eval(sess, x, y, preds_adv, mnist.test.images, mnist.test.labels, args=eval_params)
#         print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
#         #This time, we are trying to create the adversarial samples by using the FGSM approach

#         correct_label = sess.run(correct_prediction, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
#         # print("correct label " + str(correct_label))
#         correct_adv = model_evalarray(sess, x, y, preds_adv, mnist.test.images, mnist.test.labels,
#                               args=eval_params)

#         # print("correct label " + str(correct_label))
#         # print("correct adversary " + str(correct_adv))

#         for i in range(len(correct_label)):
#             if correct_label[i] == True and correct_adv[i] == True:
#                 count_adv += 1.0
#             if correct_label[i] == True:
#                 count_true += 1.0
#         # print(count_adv / count_true)
#         # util.RaiseNotDefined()

#         # print("---------------------------------------------------------")
#         # print("Creating adversarial image for " + str(source_samples) + " examples")
#         # #looping over the samples that is about to be perturbed to form the adversarial samples
#         # for sample_index in xrange(100, source_samples + 100):
#         #     current_class = int(np.argmax(mnist.test.labels[sample_index]))  
#         #     # target_class = other_classes(n_classes, current_class)

#         #     #now running the FGSM approach
#         #     adv_x = fgsm_sample(sess, x, prediction, eps = epsilon, sample=mnist.test.images[sample_index : (sample_index + 1)])
#         #     pred_adv = sess.run(prediction, feed_dict = {x : adv_x})
#         #     pred_real = sess.run(prediction, feed_dict = {x : mnist.test.images[sample_index : (sample_index + 1)]})
#         #     class_adv = int(np.argmax(pred_adv))
#         #     class_real = int(np.argmax(pred_real))
#         #     # print(adv_x.get_shape())

#         #     # if (class_real == int(np.argmax(mnist.test.labels[sample_index : (sample_index + 1)]))):
#         #     #     count_true += 1.0
#         #     #     if (class_adv == class_real):
#         #     #         count_adv += 1.0

#         #     #displaying the original and adversarial images side-by-side (only for several samples)
#         #     if 'figure' not in vars():
#         #         if class_adv == class_real:
#         #             figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
#         #                 np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' robust 020517-4 disc2 ' + str(sample_index) + '.jpg')
#         #         else:
#         #             figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
#         #                 np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' fail 020517-4 disc2 ' + str(sample_index) + ' pred ' + str(class_real) + " adv_label " + str(class_adv) + '.jpg')
#         #     else:
#         #         if class_adv == class_real:
#         #             figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
#         #                 np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' robust 020517-4 disc2 ' + str(sample_index) + '.jpg', figure=figure)
#         #         else:
#         #             figure = pair_visual(np.reshape(mnist.test.images[sample_index : (sample_index + 1)], (28, 28)),
#         #                 np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' fail 020517-4 disc2 ' + str(sample_index) + ' pred ' + str(class_real) + " adv_label " + str(class_adv) + '.jpg', figure=figure)
#         # acc_list.append(count_adv / count_true * 100)
#         print("precision rate of the  " + str(count_adv / count_true * 100))

# # plt.plot(eps_list, acc_list)
# # plt.xlabel("Epsilon")
# # plt.ylabel("Precision Adv")

# # plt.title("Accuracy for epsilon " + str(epsilon))

# # plt.savefig("170503 Exp Result eps = " + str(epsilon) + ".png")

# # plt.clf()