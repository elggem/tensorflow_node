#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

from destin import StackedAutoEncoder
from destin import SummaryWriter
from destin import OpenCVInputLayer

log.info("recording summaries to " + SummaryWriter().get_summary_folder())

model = StackedAutoEncoder(
        dims=[100],
        activations=['linear'], 
        epoch=[150],
        noise='gaussian', 
        loss='rmse',
        lr=0.007
    )

# Initialize input layer, register callback and feed video
inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)

model.register_for(inputlayer, [0,0,28,28])

inputlayer.feed_video("data/mnist.mp4", frames=20000)

model.max_activation_summary()