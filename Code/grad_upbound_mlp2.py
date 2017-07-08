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
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod, jsma, BasicIterativeMethod, LeastLikelyMethod
from cleverhans.attacks_tf import fgsm_sample, deepfool
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

"""
Method to implement : Semi-Discrete Neural Network
(the function which has the slanted and non-slanted region,
in which the slanted region has gradient gamma, while the non-
slanted one has gradient 0. Also the value is being kept to 
be between 0 and M)

Here the gradient to be used shall be gamma1 for the slanted one,
and gamma2 for the others
"""

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epoch', 30, 'Number of iterations to train')
flags.DEFINE_integer('hidden_nodes1', 256, 'Number of nodes in the first hidden layer')
flags.DEFINE_integer('hidden_nodes2', 256, 'Number of nodes in the second hidden layer')
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
flags.DEFINE_float('epsilon', 0.0, 'Rate of perturbation being applied')

# Flags related to substitute
flags.DEFINE_integer('holdout', 100, 'Test set holdout for adversary')
flags.DEFINE_integer('data_aug', 6, 'Nb of times substitute data augmented')
flags.DEFINE_integer('nb_epochs_s', 6, 'Training epochs for each substitute')
flags.DEFINE_float('lmbda', 0.2, 'Lambda in https://arxiv.org/abs/1602.02697')

n_input = 784 #for now, keep on focusing for the MNIST data
n_hidden = 128 #defining the number of nodes in the hidden layer
n_classes = 10 #remember that MNIST has 10 classes

def discrete_bound(input_raw, lower_bound, upper_bound):
    if input_raw < lower_bound:
        return lower_bound
    elif input_raw > upper_bound:
        return upper_bound
    else:
        return np.ceil(input_raw)

def discretize_image(input_list, M):
    result_list = [[discrete_bound(255 * input_list[i][j], 0, M) / 255 for j in range(len(input_list[i]))]\
        for i in range(len(input_list))]

    return result_list

def discrete_image(input_list, num_discrete):
    result_list = [[np.ceil(num_discrete * input_list[i][j]) / num_discrete for j in range(len(input_list[i]))]\
        for i in range(len(input_list))]

    return result_list

#importing the MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

M = 100
num_discrete = 5

start_time = time.time()
# train_data = discretize_image(mnist.train.images[:], M)
# val_data = discretize_image(mnist.validation.images[:], M)
# test_data = discretize_image(mnist.test.images[:], M)

train_data = discrete_image(mnist.train.images[:], num_discrete)
val_data = discrete_image(mnist.validation.images[:], num_discrete)
test_data = discrete_image(mnist.test.images[:], num_discrete)

# train_data = mnist.train.images[:]
# val_data = mnist.validation.images[:]
# test_data = mnist.test.images[:]

# print(train_data[0][:])

print("finished discretizing image in", time.time() - start_time, "seconds")

# util.RaiseNotDefined()

l2_regularize = True

#defining the graph input
#now starting the main part of the execution flow
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

act_grad_1 = FLAGS.act_grad_1
act_grad_2 = FLAGS.act_grad_2
reg_param = FLAGS.reg_param
upper_bound = FLAGS.upper_bound

#defining the activation function that should be used here
def scalar_semi_step(x_input):
    return np.clip(np.ceil(x_input), 0, upper_bound)

semi_step = np.vectorize(scalar_semi_step)

#defining its derivative
def derivative_semistep(x_input):
    x_frac = x_input % 1
    start_constant = (1 - act_grad_2)/(act_grad_1 - act_grad_2)

    if (x_input < 0 or x_input > upper_bound):
        return act_grad_2
    elif x_frac <= start_constant:
        return act_grad_1
    else:
        return act_grad_2

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
    layer_1 = tf.nn.relu(layer_1)

    layer_out = tf.add(tf.matmul(layer_1, w_out), b_out)
    layer_out = tf.nn.relu(layer_out)

    return tf.nn.softmax(layer_out)

def substitute_model(x):
    layer_1 = tf.add(tf.matmul(x, w2), b2)
    layer_1 = tf.nn.relu(layer_1)

    layer_out = tf.add(tf.matmul(layer_1, w2_out), b2_out)
    layer_out = tf.nn.relu(layer_out)

    return tf.nn.softmax(layer_out)

def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :return:
    """
    # Define TF model graph (for the black-box model)
    preds_sub = substitute_model(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, 10)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(FLAGS.data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': FLAGS.nb_epochs_s,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    verbose=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < FLAGS.data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          FLAGS.lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': FLAGS.batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return substitute_model, preds_sub


#storing weights and biases properly
w1 = tf.get_variable("w1", shape = [n_input, FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
w_out = tf.get_variable("w_out", shape = [FLAGS.hidden_nodes2, n_classes], initializer = tf.contrib.layers.xavier_initializer())

b1 = tf.get_variable("b1", shape = [FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable("b_out", shape = [n_classes], initializer = tf.contrib.layers.xavier_initializer())

w2 = tf.get_variable("w2", shape = [n_input, FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
w2_out = tf.get_variable("w2_out", shape = [FLAGS.hidden_nodes2, n_classes], initializer = tf.contrib.layers.xavier_initializer())

b2 = tf.get_variable("b2", shape = [FLAGS.hidden_nodes1], initializer = tf.contrib.layers.xavier_initializer())
b2_out = tf.get_variable("b2_out", shape = [n_classes], initializer = tf.contrib.layers.xavier_initializer())

test_image_subs = mnist.test.images[:FLAGS.holdout]
test_labels_subs = np.argmax(mnist.test.labels[:FLAGS.holdout], axis = 1)

test_image = mnist.test.images[FLAGS.holdout:]
test_labels = mnist.test.labels[FLAGS.holdout:]

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

grad1_list = [2, 4]
grad2_list = [1e-10, 1e-12]
upperbound_list = [3, 6]
eps_list = [0.0, 0.05, 0.07, 0.1, 0.3, 0.5]
# eps_iter_list = [0.01, 0.03, 0.05]

theta_list = [0.1, 0.3, 0.5, 0.7, 1.0]
gamma_list = [0.05, 0.07, 0.1]

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
# and for the 2 gradients, it will be saved in mnist_2grads_disc.ckpt and mnist_2grads.ckpt
# saver = tf.train.Saver()
saver2 = tf.train.Saver([w1, w_out, b1, b_out])
save_path = os.path.join("/tmp", "mnist_vanilla_discstep_02.ckpt")

if FLAGS.train_mode:
    for upper in upperbound_list:
        for grad_act_1 in grad1_list:
            for grad_act_2 in grad2_list:
                # for epsilon in eps_list:
                    #begin running the session of the graph
                    with tf.Session() as sess:
                        #initializing all the global variables of the graph, which is all the parameters described
                        #and also we define the value for each of the hyperparameter here
                        init = tf.global_variables_initializer()

                        sess.run(init)
                        batch_size = FLAGS.batch_size
                        act_grad_1 = grad_act_1
                        act_grad_2 = grad_act_2
                        upper_bound = upper

                        print("currently checking for the upper bound " + str(upper_bound) + " and gradient " + str(act_grad_1) + " and " + str(act_grad_2))

                        #Begin the training phase, by using only the legitimate training samples
                        for epoch in range(training_epochs):
                            # epoch_list.append(epoch + 1)
                            avg_cost = 0.
                            total_batch = int(len(train_data)/batch_size)
                            # total_batch = 2
                            # Loop over all batches

                            ptr = 0
                            for i in range(total_batch):
                                batch_x, batch_y = train_data[ptr:ptr+batch_size], mnist.train.labels[ptr:ptr+batch_size]
                                # Run optimization op (backprop) and cost op (to get loss value)
                                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                              y: batch_y})
                                # Compute average loss
                                avg_cost += c / total_batch
                                ptr += batch_size
                            # loss_list.append(avg_cost)

                            # Display logs per epoch step
                            if epoch % display_step == 0:
                                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                                    "{:.9f}".format(avg_cost))
                        print("Optimization Finished!")

                        val_acc = sess.run(accuracy, feed_dict={x: val_data, y: mnist.validation.labels})
                        print("validation accuracy is " + str(100 * val_acc) + " percent")
                        if val_acc > current_val_acc:
                            print("best validation reached at grad " + str(act_grad_1) + ", flat gradient " + str(act_grad_2) + " and upper bound " + str(upper))
                            current_val_acc = val_acc
                            saver2.save(sess, save_path)

else:
    act_grad_1 = 4
    act_grad_2 = 1e-12
    upper_bound = 6

    # for epsilon in eps_list:

    # for theta in theta_list:
    #     for gamma in gamma_list:
    for epsilon in eps_list:
        # for eps_iter in eps_iter_list:

            count_true = 0
            count_adv = 0

            with tf.Session() as sess:
                saver2.restore(sess, save_path)
                # pred = sess.run(prediction, feed_dict={x : mnist.test.images[7:8]})
                # print(pred)
                test_acc = sess.run(accuracy, feed_dict={x : test_data, y : mnist.test.labels})
                print("Test accuracy is " + str(100 * test_acc) + " percent")

                # #firstly testing towards the whole adversarial images
                print("checking for epsilon = " + str(epsilon))
                fgsm = FastGradientMethod(semi_network, sess = sess)
                fgsm_params = {'eps' : epsilon}
                adv_images = fgsm.generate(x, **fgsm_params)
                preds_adv = semi_network(adv_images)

                # adv_images = fgsm(x, prediction, eps = epsilon)
                eval_params = {'batch_size': FLAGS.batch_size}
                # X_test_adv, = batch_eval(sess, [x], [adv_images], [mnist.test.images], args=eval_params)
                # assert X_test_adv.shape[0] == 10000, X_test_adv.shape

                # Evaluate the accuracy of the MNIST model on adversarial examples
                accuracy_adv = model_eval(sess, x, y, preds_adv, test_data, mnist.test.labels, args=eval_params)
                print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
                #This time, we are trying to create the adversarial samples by using the FGSM approach

                correct_label = sess.run(correct_prediction, feed_dict = {x : test_data, y : mnist.test.labels})
                # print("correct label " + str(correct_label))
                correct_adv = model_evalarray(sess, x, y, preds_adv, test_data, mnist.test.labels,
                                      args=eval_params)

                # print("correct label " + str(correct_label))
                # print("correct adversary " + str(correct_adv))

                for i in range(len(correct_label)):
                    if correct_label[i] == True and correct_adv[i] == True:
                        count_adv += 1.0
                    if correct_label[i] == True:
                        count_true += 1.0

                print("precision rate of the adversarial towards true is " + str(count_adv / count_true * 100) + "\n")

                #now creating for the black-box approach
                # model_sub, preds_sub = train_sub(sess, x, y, prediction, test_image_subs, test_labels_subs)

                # fgsm_par = {'eps': epsilon, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
                # fgsm = FastGradientMethod(model_sub, sess=sess)

                # # Craft adversarial examples using the substitute
                # eval_params = {'batch_size': FLAGS.batch_size}
                # x_adv_sub = fgsm.generate(x, **fgsm_par)
                # preds_adv = semi_network(x_adv_sub)

                # # Evaluate the accuracy of the "black-box" model on adversarial examples
                # accuracy_adv = model_eval(sess, x, y, preds_adv, test_image, test_labels,
                #                       args=eval_params)
                # print('Test accuracy of oracle on adversarial examples generated using the substitute: ' + str(accuracy_adv))

                #now creating the JSMA approach (Jacobian-based Saliency Map Approach)
                # print("currently checking for theta " + str(theta) + " and gamma " + str(gamma))
                # persistent_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'i')
                # correct_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'i')
                # result_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'i')
                # perturb_overall = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype = 'f')

                # # grid_shape = (n_classes, n_classes, 28, 28, 1)
                # # grid_viz_data = np.zeros(grid_shape, dtype = 'f')

                # start_time = time.time()
                # jsma = SaliencyMapMethod(semi_network, back='tf', sess=sess)
                # for i in xrange(0, FLAGS.source_samples):
                #     # print("trying to make " + str(i+1)+" adversarial example")
                #     current_class = int(np.argmax(mnist.test.labels[i:(i+1)]))
                #     target_class = other_classes(FLAGS.nb_classes, current_class)

                #     for target in target_class:
                #         one_hot_target = np.zeros((1, FLAGS.nb_classes), dtype = np.float32)
                #         one_hot_target[0, target] = 1
                #         jsma_params = {'theta': theta, 'gamma': gamma,
                #                     'nb_classes': FLAGS.nb_classes, 'clip_min': 0.,
                #                     'clip_max': 1., 'targets': y,
                #                     'y_val': one_hot_target}
                #         adv_x = jsma.generate_np(test_data[i:i+1], **jsma_params)

                #         # Computer number of modified features
                #         adv_x_reshape = adv_x.reshape(-1)
                #         test_in_reshape = mnist.test.images[i].reshape(-1)
                #         nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
                #         percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

                #         # if 'figure' not in vars():
                #         #     if (model_argmax(sess, x, prediction, adv_x) == target):
                #         #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
                #         #             name = "success misclassify fig " + str(i) + " target is " + str(target))
                #         #     elif (model_argmax(sess, x, prediction, adv_x) == model_argmax(sess, x, prediction, mnist.test.images[i:i+1])):
                #         #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
                #         #             name = "success robustify fig " + str(i) + " initial is " + str(model_argmax(sess, x, prediction, mnist.test.images[i:i+1])))
                #         # else:
                #         #     if (model_argmax(sess, x, prediction, adv_x) == target):
                #         #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
                #         #             name = "success misclassify fig " + str(i) + " target is " + str(target), figure = figure)
                #         #     elif (model_argmax(sess, x, prediction, adv_x) == model_argmax(sess, x, prediction, mnist.test.images[i:i+1])):
                #         #         figure = pair_visual(np.reshape(mnist.test.images[i:i+1], (28, 28)), np.reshape(adv_x, (28, 28)),\
                #         #             name = "success robustify fig " + str(i) + " initial is " + str(model_argmax(sess, x, prediction, mnist.test.images[i:i+1])), figure = figure)

                #         #checking for the success rate
                #         res_ = model_argmax(sess, x, prediction, adv_x) == target
                #         persist_ = model_argmax(sess, x, prediction, adv_x) == model_argmax(sess, x, prediction, test_data[i:i+1])
                #         correct_ = model_argmax(sess, x, prediction, test_data[i:i+1]) == np.argmax(mnist.test.labels[i:i+1])

                #         res = int(res_)
                #         persist = int(persist_)
                #         correct_val = int(persist_ and correct_)

                #         result_overall[target, i] = res
                #         persistent_overall[target, i] = persist
                #         correct_overall[target, i] = correct_val
                #         perturb_overall[target, i] = percent_perturb

                # n_targets_tried = (n_classes - 1) * FLAGS.source_samples
                # success_rate = float(np.sum(result_overall)) / n_targets_tried
                # robust_rate = float(np.sum(persistent_overall)) / n_targets_tried
                # correct_rate = float(np.sum(correct_overall)) / n_targets_tried
                # percent_perturbed = np.mean(perturb_overall)
                # percent_perturb_succ = np.mean(perturb_overall * (result_overall == 1))

                # # print(correct_overall)
                # # print(perturb_overall * (result_overall == 1))

                # print('theta ' + str(theta) + " gamma " + str(gamma))
                # print('Execution time for adv. examples {0:.4f} seconds'.format(time.time() - start_time))
                # print('Avg. rate of successful adv. examples {0:.4f}'.format(success_rate))
                # print('Avg. rate of successful robust examples {0:.4f}'.format(robust_rate))
                # print('Avg. rate of still-correct examples {0:.4f}'.format(correct_rate))
                # print('Avg. rate of perturbed features {0:.4f}\n'.format(percent_perturbed))
                # print('Avg. rate of perturbed features for successful adversarial examples {0:.4f}'.format(percent_perturb_succ))

                #now trying to use the Basic Iterative Method
            #     print("checking for epsilon = " + str(epsilon) + " and eps_iter " + str(eps_iter))
            #     least_like = BasicIterativeMethod(semi_network, sess = sess)
            #     basic_params = {'eps' : epsilon, 'eps_iter' : eps_iter, 'ord' : np.inf}
            #     adv_images = least_like.generate(x, **basic_params)
            #     preds_adv = semi_network(adv_images)

            # #     # firstly testing towards the whole adversarial images
            # #     # print("checking for epsilon = " + str(epsilon))
            # #     # fgsm = FastGradientMethod(semi_network, sess = sess)
            # #     # fgsm_params = {'eps' : epsilon}
            # #     # adv_images = fgsm.generate(x, **fgsm_params)
            # #     # preds_adv = semi_network(adv_images)

            #     # adv_images = fgsm(x, prediction, eps = epsilon)
            #     eval_params = {'batch_size': FLAGS.batch_size}
            #     # X_test_adv, = batch_eval(sess, [x], [adv_images], [mnist.test.images], args=eval_params)
            #     # assert X_test_adv.shape[0] == 10000, X_test_adv.shape

            #     # Evaluate the accuracy of the MNIST model on adversarial examples
            #     accuracy_adv = model_eval(sess, x, y, preds_adv, mnist.test.images, mnist.test.labels, args=eval_params)
            #     print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
            # # # #This time, we are trying to create the adversarial samples by using the FGSM approach

            #     correct_label = sess.run(correct_prediction, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
            #     # print("correct label " + str(correct_label))
            #     correct_adv = model_evalarray(sess, x, y, preds_adv, mnist.test.images, mnist.test.labels,
            #                           args=eval_params)

            #     # print("correct label " + str(correct_label))
            #     # print("correct adversary " + str(correct_adv))

            #     for i in range(len(correct_label)):
            #         if correct_label[i] == True and correct_adv[i] == True:
            #             count_adv += 1.0
            #         if correct_label[i] == True:
            #             count_true += 1.0

            #     print("precision rate of the adversarial towards true is " + str(count_adv / count_true * 100) + "\n")

            # print("now trying to use adversarial training")
            # init2 = tf.variables_initializer([w1, b1, w_out, b_out])
            # sess.run(init2)
            # print("variables initialized")

            # fgsm2 = FastGradientMethod(substitute_model, sess=sess)
            # prediction2_adv = semi_network(fgsm2.generate(x, **fgsm_params))

            # def evaluate_2():
            #     accuracy_2 = model_eval(sess, x, y, semi_network(x), mnist.test.images, mnist.test.labels, args=eval_params)
            #     print('Test accuracy on legitimate test examples: %0.4f' % accuracy_2)

            #     accuracy_2_adv = model_eval(sess, x, y, prediction2_adv, mnist.test.images, mnist.test.labels, args=eval_params)
            #     print('Test accuracy on adversarial examples: %0.4f' % accuracy_2_adv)

            # model_train(sess, x, y, semi_network(x), mnist.train.images, mnist.train.labels,\
            #     predictions_adv = prediction2_adv, evaluate = evaluate_2, args = train_params,\
            #     w1 = w1, w_out = w_out, b1 = b1, b_out = b_out, reg_param = reg_param)


            # correct_label = sess.run(correct_prediction, feed_dict = {x : mnist.test.images, y : mnist.test.labels})
            # # print("correct label " + str(correct_label))
            # correct_adv = model_evalarray(sess, x, y, preds_adv, mnist.test.images, mnist.test.labels,
            #                       args=eval_params)

            # # print("correct label " + str(correct_label))
            # # print("correct adversary " + str(correct_adv))

            # for i in range(len(correct_label)):
            #     if correct_label[i] == True and correct_adv[i] == True:
            #         count_adv += 1.0
            #     if correct_label[i] == True:
            #         count_true += 1.0
            # print(count_adv / count_true)
            # # # util.RaiseNotDefined()

            # print("---------------------------------------------------------")
            # print("Creating adversarial image for " + str(FLAGS.source_samples) + " examples")
            # #looping over the samples that is about to be perturbed to form the adversarial samples
                for sample_index in xrange(100, FLAGS.source_samples + 100):
                    current_class = int(np.argmax(mnist.test.labels[sample_index]))  
                    # target_class = other_classes(n_classes, current_class)

                    #now running the FGSM approach
                    adv_x = fgsm_sample(sess, x, prediction, eps = epsilon, sample=test_data[sample_index : (sample_index + 1)])
                    pred_adv = sess.run(prediction, feed_dict = {x : adv_x})
                    pred_real = sess.run(prediction, feed_dict = {x : test_data[sample_index : (sample_index + 1)]})
                    class_adv = int(np.argmax(pred_adv))
                    class_real = int(np.argmax(pred_real))
                    # print(adv_x.get_shape())

                    # if (class_real == int(np.argmax(mnist.test.labels[sample_index : (sample_index + 1)]))):
                    #     count_true += 1.0
                    #     if (class_adv == class_real):
                    #         count_adv += 1.0

                    #displaying the original and adversarial images side-by-side (only for several samples)
                    if 'figure' not in vars():
                        if class_adv == class_real:
                            figure = pair_visual(np.reshape(test_data[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' robust 050717 vanilla_discrete_02 ' + str(sample_index) + '.jpg')
                        else:
                            figure = pair_visual(np.reshape(test_data[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' fail 050717 vanilla_discrete_02 ' + str(sample_index) + ' pred ' + str(class_real) + " adv_label " + str(class_adv) + '.jpg')
                    else:
                        if class_adv == class_real:
                            figure = pair_visual(np.reshape(test_data[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' robust 050717 vanilla_discrete_02 ' + str(sample_index) + '.jpg', figure=figure)
                        else:
                            figure = pair_visual(np.reshape(test_data[sample_index : (sample_index + 1)], (28, 28)),
                                np.reshape(adv_x, (28, 28)), name = 'eps ' + str(epsilon) + ' fail 050717 vanilla_discrete_02 ' + str(sample_index) + ' pred ' + str(class_real) + " adv_label " + str(class_adv) + '.jpg', figure=figure)
            # acc_list.append(count_adv / count_true * 100)

# # plt.plot(eps_list, acc_list)
# # plt.xlabel("Epsilon")
# # plt.ylabel("Precision Adv")

# # plt.title("Accuracy for epsilon " + str(epsilon))

# # plt.savefig("170503 Exp Result eps = " + str(epsilon) + ".png")

# # plt.clf()
