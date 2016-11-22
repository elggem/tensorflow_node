#!/usr/bin/env python
# -*- coding: utf-8 -*-

from code import StackedAutoEncoder
from code import SummaryWriter
from code import OpenCVInputLayer
from code import log

log.info("recording summaries to " + SummaryWriter().get_summary_folder())

model = StackedAutoEncoder(
        dims=[100],
        encoding_activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007,
        metadata=True,
        timeline=True
    )

# Initialize input layer, register callback and feed video
inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=5)

model.register_for_inputlayer(inputlayer, [0,0,28,28])

inputlayer.feed_video("data/mnist.mp4", frames=20)

model.max_activation_summary()
model.save_parameters()