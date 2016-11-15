#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from sklearn import datasets
from model import StackedAutoEncoder
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

import numpy as np
import model.utils as utils
from os.path import join as pjoin

#utils.start_tensorboard()

train_data = mnist.train.images 

print "👉 processed input data!"

models = []

global_session = tf.Session()

for _ in xrange(4):
    models.append(StackedAutoEncoder(
        dims=[100],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[50],
        loss='rmse',
        lr=0.007,
        batch_size=150,
        print_step=50,
        session = global_session
    ))

for i in xrange(3):
    for model in models:
        model.fit(train_data)
        model.save_weights()
        
