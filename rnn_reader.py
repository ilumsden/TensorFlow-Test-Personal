from __future__ import absolute_import, division, print_function

import collections
import os
import sys

import tensorflow as tf

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if sys.version_info[0] >= 3:
            return(f.read().replace("\n", "<eos>").split()
        else:
            return(f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id
