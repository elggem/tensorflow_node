#!/usr/bin/env python
# -*- coding: utf-8 -*-

from code import StackedAutoEncoder
from code import SummaryWriter
from code import OpenCVInputLayer
from code import log

log.info("recording summaries to " + SummaryWriter().directory)

models = []

for _ in xrange(4):
    models.append(StackedAutoEncoder(
        dims=[100],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[10],
        loss='rmse',
        lr=0.007
    ))

# Initialize input layer, register callback and feed video
inputlayer = OpenCVInputLayer(output_size=(32,32), batch_size=1)

models[0].register_for_inputlayer(inputlayer, [00,00,16,16])
models[1].register_for_inputlayer(inputlayer, [16,00,16,16])
models[2].register_for_inputlayer(inputlayer, [00,16,16,16])
models[3].register_for_inputlayer(inputlayer, [16,16,16,16])

inputlayer.feed_video("data/hand.m4v", frames=20)

for model in models:
    model.max_activation_summary()
