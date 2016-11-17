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
        dims=[1000],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[150],
        loss='rmse',
        lr=0.007
    )

# Initialize input layer, register callback and feed video
inputlayer = OpenCVInputLayer(output_size=(160,160), batch_size=30)

inputlayer.registerCallback([0,0,160,160], model.fit)

inputlayer.feedVideo("data/hand.m4v")

model.write_activation_summary()