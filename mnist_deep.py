from __future__ import absolute_import, division, print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def deepnn(x):
    x_image = tf.reshape(x, [-1,28,28,1])
    #First Convolutional Layer
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    #First Pooling Layer
    h_pool1 = max_pool_2x2(h_conv1)
    #Seconds Convolutional Layer
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #Second Pooling Layer
    h_pool2 = max_pool_2x2(h_conv2)
    #Fully Connected Layer 1
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #Map the 1024 features to 10 classes (1/digit)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y, keep_prob

def main(_):
    #Import Data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    #Create and Model
    x = tf.placeholder(tf.float32, [None, 784])
    corr = tf.placeholder(tf.float32, shape=[None, 10])
    #Build the graph for the Deep Net
    y_conv, keep_prob = deepnn(x)

    cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                               (labels=corr, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_ent)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(corr,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], 
                                     corr: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, 
                       train_accuracy))
            train_step.run(feed_dict={x: batch[0], corr: batch[1], 
                            keep_prob: 0.5})
        print("test accuracy %g" %accuracy.eval(feed_dict={x: 
             mnist.test.images, corr: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        default="/tmp/tensorflow/mnist/input_data", 
                        help="Directory for storing input data")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

#Don't run on laptop! There are 20,000 intensive training loops. Consider using GPU. Should run everything on GPU that can run on GPU by default.
