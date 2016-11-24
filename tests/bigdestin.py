#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

from destin import StackedAutoEncoder
from destin import SummaryWriter
from destin import OpenCVInputLayer

log.info("recording summaries to " + SummaryWriter().get_summary_folder())

model_input_a = StackedAutoEncoder(
        name="ae-a",
        dims=[16],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_input_b = StackedAutoEncoder(
        name="ae-b",
        dims=[16],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_input_c = StackedAutoEncoder(
        name="ae-c",
        dims=[16],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_input_d = StackedAutoEncoder(
        name="ae-d",
        dims=[16],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )


model_middle_ab = StackedAutoEncoder(
        name="ae-m-ab",
        dims=[16],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_middle_cd = StackedAutoEncoder(
        name="ae-m-cd",
        dims=[16],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_top = StackedAutoEncoder(
        name="ae-top",
        dims=[16],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

# Initialize input layer, register callback and feed video
inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)


#connect a-d to corners of input, no stride
model_input_a.register_for(inputlayer, [00,00,14,14])
model_input_b.register_for(inputlayer, [14,00,14,14])
model_input_c.register_for(inputlayer, [00,14,14,14])
model_input_d.register_for(inputlayer, [14,14,14,14])

#connect middle layers to input layers
model_middle_ab.register_for(model_input_a)
model_middle_ab.register_for(model_input_b)
model_middle_cd.register_for(model_input_c)
model_middle_cd.register_for(model_input_d)

#connect middle layers to top layer
model_top.register_for(model_middle_ab)
model_top.register_for(model_middle_cd)

inputlayer.feed_video("data/hand.m4v")

model_top.max_activation_recursive()
