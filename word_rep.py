"""URL for preprocessing code: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/word2vec/word2vec_basic.py"""

from __future__ import absolute_import, division, print_function

import collections
import math
import os
import random
import zipfile
import urllib

import numpy as np
#from six.moves import urllib
import tensorflow as tf

#Step 1: Data Download and String Conversion
#url for Data Download
url = "http://mattmahoney.net/dc/"

def maybe_download(filename, expected_bytes):
    """Download a file if it's not present, and validate its size"""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = of.stat(filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified ", filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "Failed to verify " + filename + 
            ". Can you get to it with a browser?")
    return filename

filename = maybe_download("text8.zip", 31344016)

def read_data(filename):
    """Extract the first file of a zip as a list of words"""
    with zipfile.Zipfile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print("Data size", len(words))

#Step 2: Build the dictionary for learning and replace rare words with UNK 
#token.
vocabulary_size = 50000

def build_dataset(words, vocabulary_size):
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, 
                                                            vocabulary_size)

del words #Done to reduce memory usage

print("Most common words (+UNK)", count[:5])
print("Sample Data ", data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

#Step 3: Generate a training batch for the Skip-Gram model
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
