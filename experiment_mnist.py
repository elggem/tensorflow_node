#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import StackedAutoEncoder
from model import SummaryWriter
from inputlayer import OpenCVInputLayer

import tensorflow as tf

import numpy as np
import model.utils as utils
from os.path import join as pjoin

#utils.start_tensorboard()

print "recording summaries to " + SummaryWriter().directory

model = StackedAutoEncoder(
        dims=[100],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[50],
        loss='rmse',
        lr=0.007
    )

# Initialize input layer, register callback and feed video
inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)

inputlayer.registerCallback([0,0,28,28], model.fit_transform)

inputlayer.feedVideo("data/mnist.mp4", frames=30000)

model.write_activation_summary()