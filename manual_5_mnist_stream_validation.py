#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import StackedAutoEncoder
from inputlayer import OpenCVInputLayer

import tensorflow as tf

import numpy as np
import model.utils as utils
from os.path import join as pjoin

#utils.start_tensorboard()

print "recording summaries to " + utils.get_summary_dir()

model = StackedAutoEncoder(
        dims=[100],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[1],
        loss='rmse',
        lr=0.007,
        batch_size=1
    )

### Initialize Enqueue thread:
inputlayer = OpenCVInputLayer()

inputlayer.registerCallback([0,0,28,28], model.fit_single)

inputlayer.feedVideo("data/mnist.mp4", frames=5000)

model.write_activation_summary()
