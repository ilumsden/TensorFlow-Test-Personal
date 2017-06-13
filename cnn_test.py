#Note: this ML algorithm has 20,000 learning cycles. Recomend evaluating on a computer with GPU.

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as mfn

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for a CNN."""
    #input layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    #Convolutional layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    #Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    #Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    #Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    #Dense Layer 1
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, 
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, 
                                training=mode == learn.ModeKeys.TRAIN)

    #Logits Layer (aka Dense Layer 2)
    logits = tf.layers.dense(inputs=dropout, units=10)
    loss = None
    train_op = None

    #Calculate Loss (for TRAIN AND EVAL)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), 
                                   depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
                                               logits=logits)

    #Configure Training Op
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss, 
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

    #Generate Predictions
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, 
                       name="softmax_tensor")}

    #Return a ModelFnOps object
    return mfn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, 
                          train_op=train_op)

def main(unused_argv):
    #Load Test and Evaluation Data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images #this is a numpy array
    train_labels = np.asarray(mnist.train.lables, dtype=np.int32)
    eval_data = mnist.test.images #this in a numpy array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    #Create the Estimator
    mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, 
                                       model_dir="/tmp/mnist_convnet_mode1")

    #Set up prediction logging
    log_tensors = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=log_tensors, 
                                              every_n_iter=50)

    #Model Training
    mnist_classifier.fit(x=train_data, y=train_labels, batch_size=100, 
                         steps=20000, monitors=[logging_hook])

    #Configure the accuracy metric for evaluation
    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, 
               prediction_key="classes"),}

    #Evaluate and print the results
    eval_results = mnist_classifier.evaluate(x=eval_data, y=eval_labels, 
                                             metrics=metrics)
    print(eval_results)
