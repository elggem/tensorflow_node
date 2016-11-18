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

inputlayer.registerCallback([00,00,16,16], models[0].fit)
inputlayer.registerCallback([16,00,16,16], models[1].fit)
inputlayer.registerCallback([00,16,16,16], models[2].fit)
inputlayer.registerCallback([16,16,16,16], models[3].fit)

inputlayer.feedVideo("data/hand.m4v", frames=20)

for model in models:
    model.max_activation_summary()
