import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
import os
import time

from tensorflow.examples.tutorials.mnist import input_data
from keras.utils.np_utils import to_categorical
from tensorflow.python.framework import ops
from PIL import Image
from six.moves import xrange

from cleverhans.utils import other_classes, pair_visual, grid_visual
from cleverhans.utils_tf import model_train, model_eval, batch_eval, model_loss, model_evalarray, model_argmax
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod, jsma, BasicIterativeMethod
from cleverhans.attacks_tf import fgsm_sample

"""
Method to implement : DeepCloak (reproducing the results from
https://arxiv.org/abs/1702.06763)
"""

#the flags being defined
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epoch', 30, 'Number of iterations to train')
flags.DEFINE_integer('hidden_nodes1', 128, 'Number of nodes in the first hidden layer')
flags.DEFINE_integer('hidden_nodes2', 128, 'Number of nodes in the second hidden layer')
flags.DEFINE_integer('batch_size', 128, 'Size of the batch')
flags.DEFINE_float('learning_rate', 0.001, 'Rate of learning for the optimizer')
# flags.DEFINE_float('dropout_rate', 0.2, 'Rate of dropout')
flags.DEFINE_boolean('l2Regularizer', True, 'whether we apply l2Regularize or not')
flags.DEFINE_float('reg_param', 0.1, "parameter to measure the power of regularizer")
flags.DEFINE_integer('act_grad_1', 2, 'the gradient of the slanted region')
flags.DEFINE_float('act_grad_2', 1e-10, 'the gradient of the non-slanted region')
flags.DEFINE_integer('upper_bound', 10, 'the upper bound of the value of activation fct')
flags.DEFINE_integer('nb_classes', 10, 'number of classes available in the dataset')
flags.DEFINE_integer('source_samples', 50, 'number of samples to be tested')
flags.DEFINE_boolean('train_mode', True, 'determiner on whether we run the training or testing mode')

flags.DEFINE_integer('feature_remove', 8, "number of features to be removed")

n_input = 784
n_hidden = 256
n_classes = 10
reg_param = FLAGS.reg_param
training_epochs = FLAGS.training_epoch
display_step = 1

#importing the data from MNIST example
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

l2_regularize = True
eps_list = [0.0, 0.05, 0.07, 0.1, 0.3, 0.5]

theta_list = [0.3, 0.5, 1.0]
gamma_list = [0.05, 0.1]

#defining the graph input
#now starting the main part of the execution flow
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
keep_prob = tf.placeholder("float")
mask = tf.placeholder("float", [256])

# storing weights and biases properly
w1 = tf.get_variable("w1", shape = [n_input, n_hidden], initializer = tf.contrib.layers.xavier_initializer())
w_out = tf.get_variable("w_out", shape = [n_hidden, n_classes], initializer = tf.contrib.layers.xavier_initializer())

b1 = tf.get_variable("b1", shape = [n_hidden], initializer = tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable("b_out", shape = [n_classes], initializer = tf.contrib.layers.xavier_initializer())

# wc1 = tf.Variable(tf.random_normal([5,5,1,16]))
# wd1 = tf.Variable(tf.random_normal([7*7*16, 10]))

# bc1 = tf.Variable(tf.random_normal([16]))

saver = tf.train.Saver()
save_path = os.path.join("/tmp", "mnist_vanilla.ckpt")

def conv_2d(x, w, b, strides=1):
	conv_layer = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
	conv_layer = tf.nn.bias_add(conv_layer, b)
	return tf.nn.relu(conv_layer)

def maxpool2d(x, k=4):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides = [1,k,k,1], padding = 'SAME')

def conv_net(x):
	layer_1 = tf.add(tf.matmul(x, w1), b1)
	layer_1 = tf.nn.relu(layer_1)

	layer_out = tf.add(tf.matmul(layer_1, w_out), b_out)
	layer_out = tf.nn.relu(layer_out)

	return tf.nn.softmax(layer_out)


# def mlp_network(x):
# 	layer_1 = tf.add(tf.matmul(x, w1), b1)
# 	layer_1 = tf.nn.relu(layer_1)

# 	layer_out = tf.add(tf.matmul(layer_1, w_out), b_out)
# 	layer_out = tf.nn.relu(layer_out)

# 	return tf.nn.softmax(layer_out)

def feature_network(x):
	layer_1 = tf.add(tf.matmul(x, w1), b1)
	layer_1 = tf.nn.relu(layer_1)

	return layer_1

# def cnn_deepcloak(x, mask):
# 	x = tf.reshape(x, shape = [-1, 28, 28, 1])

# 	conv1 = conv_2d(x, wc1, bc1)
# 	conv1 = maxpool2d(conv1)

# 	#start to define the fully connected layer
# 	fc1 = tf.reshape(conv1, [-1, wd1.get_shape().as_list()[0]])
# 	fc1 = tf.multiply(fc1, mask)
# 	fc1 = tf.matmul(fc1, wd1)
# 	fc1 = tf.nn.relu(fc1)

# 	return tf.nn.softmax(fc1)

def mlp_deepcloak(x, mask):
	layer_1 = tf.add(tf.matmul(x, w1), b1)
	layer_1 = tf.nn.relu(layer_1)
	layer_1 = tf.multiply(layer_1, mask)

	layer_out = tf.add(tf.matmul(layer_1, w_out), b_out)
	layer_out = tf.nn.relu(layer_out)

	return tf.nn.softmax(layer_out)

#constructing the model
prediction = conv_net(x)
feature_val = feature_network(x)
masked_pred = mlp_deepcloak(x, mask)

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
    mask_correct = tf.equal(tf.argmax(masked_pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc_mask = tf.reduce_mean(tf.cast(mask_correct, "float"))

if FLAGS.train_mode:
	print("Entering the training phase")
	with tf.Session() as sess:
		#initializing all the global variables of the graph, which is all the parameters described
		#and also we define the value for each of the hyperparameter here
		init = tf.global_variables_initializer()

		sess.run(init)
		batch_size = FLAGS.batch_size

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
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
	        	# Compute average loss
				avg_cost += c / total_batch
	        # loss_list.append(avg_cost)

	        # Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
		print("Optimization Finished!")

		real_acc = sess.run(accuracy, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
		print("the accuracy for real dataset is " + str(real_acc))

		saver.save(sess, save_path)
		print("Parameter saved")

else:
	print("Entering the testing phase")
	# for epsilon in eps_list:
	with tf.Session() as sess:
		saver.restore(sess, save_path)
		real_acc = sess.run(accuracy, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
		print("the accuracy for real dataset is " + str(real_acc))
		for theta in theta_list:
			for gamma in gamma_list:
				for feature_remove in [4]:

					count_true = 0
					count_adv = 0
					# print("testing with epsilon " + str(epsilon))


					
					# #first : creating the adversarial image by using FGSM
					# # basic_iter = BasicIterativeMethod(conv_net, sess=sess)
					# fgsm = BasicIterativeMethod(conv_net, sess = sess)
					# # fgsm_params = {'eps' : epsilon}
					# fgsm_params = {'eps' : epsilon, 'eps_iter' : eps_iter}
					# adv_images = fgsm.generate(x, **fgsm_params)
					# preds_adv = conv_net(adv_images)

					# eval_params = {'batch_size': FLAGS.batch_size}

					# accuracy_adv = model_eval(sess, x, y, preds_adv, mnist.test.images, mnist.test.labels, args=eval_params)
					# print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

					# print('epsilon ' + str(epsilon) + ' eps_iter ' + str(eps_iter) + ' feature remove ' + str(feature_remove))

					# X_test_adv, = batch_eval(sess, [x], [adv_images], [mnist.test.images], args=eval_params)
					# # adversary_acc = sess.run(accuracy, feed_dict = {x : X_test_adv, y : mnist.test.labels})

					# #second : removing several features by doing masking
					# adv_feature = sess.run(feature_val, feed_dict = {x : X_test_adv})
					# img_feature = sess.run(feature_val, feed_dict = {x : mnist.test.images})
					# diff = tf.abs(tf.subtract(adv_feature, img_feature))
					# diff = tf.reduce_sum(diff, 0)
					# # diff = tf.Print(diff, [diff], summarize=84)
					# mask_ = np.ones([256])

					# #finding the index of elements with highest value
					# _, indices = tf.nn.top_k(diff, feature_remove)
					# # indices = tf.Print(indices, [indices], summarize = 7)
					# for index in indices.eval():
					# 	mask_[index] = 0
					# # # print(mask_)

					# preds_mask_adv = mlp_deepcloak(adv_images, mask_)
					# correct_label = sess.run(correct_prediction, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
					# correct_adv = model_evalarray(sess, x, y, preds_mask_adv, mnist.test.images, mnist.test.labels, args=eval_params)


					# # correct_label = sess.run(correct_prediction, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
					# #                 # print("correct label " + str(correct_label))
					# # correct_adv = model_evalarray(sess, x, y, preds_adv, mnist.test.images, mnist.test.labels,
					# #                                       args=eval_params)

					# for i in range(len(correct_label)):
					# 	if correct_label[i] == True and correct_adv[i] == True:
					# 		count_adv += 1.0
					# 	if correct_label[i] == True:
					# 		count_true += 1.0
					# print("precision rate of the adversarial towards true is " + str(count_adv / count_true * 100))

					# #now testing the performance towards the adversarial examples
					# accuracy_mask = sess.run(acc_mask, feed_dict = {x : X_test_adv, y : mnist.test.labels, mask : mask_})
					# # print("the accuracy for real dataset is " + str(real_acc))
					# # print("the accuracy for the adversarial examples is " + str(adversary_acc))
					# print("the accuracy of the adversarial examples in masked is " + str(accuracy_mask) + "\n")

					#now creating the JSMA approach (Jacobian-based Saliency Map Approach)
					print("currently checking for theta " + str(theta) + " gamma " + str(gamma) + " feature removal " + str(feature_remove))
					persistent_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'i')
					correct_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'i')
					result_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'i')
					perturb_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'f')

					# grid_shape = (n_classes, n_classes, 28, 28, 1)
					# grid_viz_data = np.zeros(grid_shape, dtype = 'f')

					start_time = time.time()
					jsma = SaliencyMapMethod(conv_net, back='tf', sess=sess)
					for i in xrange(0, FLAGS.source_samples):
					    # print("trying to make " + str(i+1)+" adversarial example")
						current_class = int(np.argmax(mnist.test.labels[i:(i+1)]))
						target_class = other_classes(FLAGS.nb_classes, current_class)

						for target in target_class:
							one_hot_target = np.zeros((1, FLAGS.nb_classes), dtype = np.float32)
							one_hot_target[0, target] = 1
							jsma_params = {'theta': theta, 'gamma': gamma,
					                    'nb_classes': FLAGS.nb_classes, 'clip_min': 0.,
					                    'clip_max': 1., 'targets': y,
					                    'y_val': one_hot_target}
							adv_x = jsma.generate_np(mnist.test.images[i:i+1], **jsma_params)

					        # Computer number of modified features
							adv_x_reshape = adv_x.reshape(-1)
							test_in_reshape = mnist.test.images[i].reshape(-1)
							nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
							percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

							#second : removing several features by doing masking
							adv_feature = sess.run(feature_val, feed_dict = {x : adv_x})
							img_feature = sess.run(feature_val, feed_dict = {x : mnist.test.images})
							diff = tf.abs(tf.subtract(adv_feature, img_feature))
							diff = tf.reduce_sum(diff, 0)
					        # diff = tf.Print(diff, [diff], summarize=84)
							mask_ = np.ones([256])

					        #finding the index of elements with highest value
							_, indices = tf.nn.top_k(diff, feature_remove)
					        # indices = tf.Print(indices, [indices], summarize = 7)
							for index in indices.eval():
								mask_[index] = 0
					        # # print(mask_)

							preds_mask_adv = mlp_deepcloak(adv_x, mask_)

					        # if 'figure' not in vars():
					        #     if (model_argmax(sess, x, prediction, adv_x) == target):
					        #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
					        #             name = "success misclassify fig " + str(i) + " target is " + str(target))
					        #     elif (model_argmax(sess, x, prediction, adv_x) == model_argmax(sess, x, prediction, mnist.test.images[i:i+1])):
					        #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
					        #             name = "success robustify fig " + str(i) + " initial is " + str(model_argmax(sess, x, prediction, mnist.test.images[i:i+1])))
					        # else:
					        #     if (model_argmax(sess, x, prediction, adv_x) == target):
					        #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
					        #             name = "success misclassify fig " + str(i) + " target is " + str(target), figure = figure)
					        #     elif (model_argmax(sess, x, prediction, adv_x) == model_argmax(sess, x, prediction, mnist.test.images[i:i+1])):
					        #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
					        #             name = "success robustify fig " + str(i) + " initial is " + str(model_argmax(sess, x, prediction, mnist.test.images[i:i+1])), figure = figure)

					        #checking for the success rate
							res_ = model_argmax(sess, x, preds_mask_adv, adv_x) == target
							persist_ = model_argmax(sess, x, preds_mask_adv, adv_x) == model_argmax(sess, x, preds_mask_adv, mnist.test.images[i:i+1])
							correct_ = model_argmax(sess, x, preds_mask_adv, mnist.test.images[i:i+1]) == np.argmax(mnist.test.labels[i:i+1])

							res = int(res_)
							persist = int(persist_)
							correct_val = int(persist_ and correct_)

							result_overall[target, i] = res
							persistent_overall[target, i] = persist
							correct_overall[target, i] = correct_val
							perturb_overall[target, i] = percent_perturb

					n_targets_tried = (n_classes - 1) * FLAGS.source_samples
					success_rate = float(np.sum(result_overall)) / n_targets_tried
					robust_rate = float(np.sum(persistent_overall)) / n_targets_tried
					correct_rate = float(np.sum(correct_overall)) / n_targets_tried
					percent_perturbed = np.mean(perturb_overall)
					percent_perturb_succ = np.mean(perturb_overall * (result_overall == 1))

					# print(correct_overall)
					# print(perturb_overall * (result_overall == 1))

					print('theta ' + str(theta) + " gamma " + str(gamma))
					print('Execution time for adv. examples {0:.4f} seconds'.format(time.time() - start_time))
					print('Avg. rate of successful adv. examples {0:.4f}'.format(success_rate))
					print('Avg. rate of successful robust examples {0:.4f}'.format(robust_rate))
					print('Avg. rate of still-correct examples {0:.4f}'.format(correct_rate))
					print('Avg. rate of perturbed features {0:.4f}\n'.format(percent_perturbed))