#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from sklearn import datasets
from model import StackedAutoEncoder

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

import numpy as np
import model.utils as utils
from os.path import join as pjoin

#mnist.train.images.shape = (55000, 784)
#mnist.test.images.shape = (10000, 784)

"""
padded_train_data = np.ndarray([mnist.train.images.shape[0],1024], dtype='float32')

for i, image in enumerate(mnist.train.images):
    padded_train_data[i] = np.pad(image.reshape([28,28]),2,'constant').flatten()
"""

padded_train_data = mnist.train.images ## 748
#padded_train_data.shape == (55000,1024)

"""
padded_train_data = np.ndarray([mnist.train.images.shape[0],mnist.train.images.shape[1]], dtype='float32')

for i, image in enumerate(mnist.train.images):
    padded_train_data[i] = image/np.sum(image)
"""

print "👉 processed input data!"

#iris = datasets.load_iris().data

model = StackedAutoEncoder(
    dims=[100],
    activations=['linear'], 
    noise='gaussian', 
    epoch=[50],
    loss='rmse',
    lr=0.007,
    batch_size=150
)

for i in xrange(80):
    model.fit(padded_train_data)
    utils.plot_max_activation_fast(model, "epoch_%04d_linear_0.5_gaussian.png" % (i*50))
