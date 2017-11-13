# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride[1,x_movement,y_movement,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


## define placeholder for inputs to network##
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # [n_samples, 28, 28, 1]

## define conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 20])  # patch 5x5, in size 1, out size 20
b_conv1 = bias_variable([20])
h_conv1 = conv2d(x_image, W_conv1) + b_conv1  # output size 20
h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*20

## define conv2 layer ##
W_conv2 = weight_variable([5, 5, 20, 50])  # patch 5x5, in size 20, out size 50
b_conv2 = bias_variable([50])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2  # output size 14x14x50
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x50

##function layer1##
W_fc1 = weight_variable([7 * 7 * 50, 500])
b_fc1 = bias_variable([500])
# [n_samples,7,7,50]  ->> [n_samples,7*7*50]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 50])

h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##function layer2##
W_fc2 = weight_variable([500, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# define Session()
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 1.0})
        print("step %d, train accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 1.0})

    if i % 1000 == 0:
        print("############step %d, test accuracy %g" % (i, accuracy.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0})))

