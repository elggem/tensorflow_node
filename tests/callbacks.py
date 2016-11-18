#!/usr/bin/env python
# -*- coding: utf-8 -*-

from code import StackedAutoEncoder
from code import SummaryWriter
from code import OpenCVInputLayer
from code import log

log.info("recording summaries to " + SummaryWriter().directory)

model_input = StackedAutoEncoder(
        name="ae-input",
        dims=[50],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_top = StackedAutoEncoder(
        name="ae-top",
        dims=[100],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

# Initialize input layer, register callback and feed video
inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=5)

inputlayer.registerCallback([0,0,28,28], model_input.fit_transform)

model_input.registerCallback(model_top.fit_transform)

inputlayer.feedVideo("data/mnist.mp4", frames=100)

#model.max_activation_summary()
#model.save_parameters()