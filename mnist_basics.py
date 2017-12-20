#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Rock
# @File    : mnist_basics.py
# @Time    : 12/19/2017 9:55 AM

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=True)

# print the size of training/val/test set
print("Training data size: ", mnist.train.num_examples)
print("Validation data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)

# print an example of a training image
print("Format of a training image: ", mnist.train.images[0])

# print an example of a training image's label
print("The label of a training image: ", mnist.train.labels[0])

