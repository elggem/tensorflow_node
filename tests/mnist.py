#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from code import StackedAutoEncoder
from code import SummaryWriter
from code import OpenCVInputLayer

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

model.max_activation_summary()