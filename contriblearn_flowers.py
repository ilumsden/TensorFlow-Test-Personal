from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

#Sets Logging to the Info severity level
tf.logging.set_verbosity(tf.logging.INFO)

#Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_training.csv"

def get_train_inputs():
    #Define the training inputs
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x,y

def get_test_inputs():
    #Define the test inputs
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x,y

def new_samples():
    #Classify 2 new flower samples.
    return np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

def main():
    #Download the training and test sets if not stored locally
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "w") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "w") as f:
            f.write(raw)

    #Load Datasets
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING, target_dtype=np.int, 
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)

    #Specify that all the features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    #Creates a validation monitor to monitor the learning
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_set.data, test_set.target, every_n_steps=50)

    #Build a 3-layer DNN with 10, 20, 10 neurons respectively
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[10,20,10], 
        n_classes=3, model_dir="/tmp/iris_model", 
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    #Fit the model
    classifier.fit(x=training_set.data, y=training_set.target, steps=2000,
                   monitors=[validation_monitor])

    #Evaluate the model's accuracy and print the result.
    accuracy = classifier.evaluate(x=test_set.data, y=test_set.target,
                                   steps=1)["accuracy"]

    print("Test Accuracy: {0:f}".format(accuracy))


    predictions = list(classifier.predict(input_fn=new_samples, 
                       as_iterable=True))

    print(
        "Predictions:  {}".format(str(predictions)))

    #for p in predictions:
        #ind = predictions.index(p)
        #if p == 0:
            #print("Flower " + str(ind) + " is Iris setosa.")
        #elif p == 1:
            #print("Flower " + str(ind) + " is Iris versicolor.")
        #else:
            #print("Flower " + str(ind) + " is Iris virginica.")

if __name__ == "__main__":
    main()


