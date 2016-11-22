#!/usr/bin/env python
# -*- coding: utf-8 -*-

from code import StackedAutoEncoder
from code import SummaryWriter
from code import OpenCVInputLayer
from code import log

log.info("recording summaries to " + SummaryWriter().get_summary_folder())

model_input_a = StackedAutoEncoder(
        name="ae-input-A",
        dims=[100],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_input_b = StackedAutoEncoder(
        name="ae-input-B",
        dims=[200],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_input_c = StackedAutoEncoder(
        name="ae-input-C",
        dims=[20],
        activations=['linear'], 
        noise='gaussian', 
        epoch=[100],
        loss='rmse',
        lr=0.007
    )

model_input_d = StackedAutoEncoder(
        name="ae-input-D",
        dims=[30],
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
inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)

#connect a to left and b to right side of input.
model_input_a.register_for_inputlayer(inputlayer, [07,07,14,14])
model_input_b.register_for_inputlayer(inputlayer, [00,00,28,28])
model_input_c.register_for_inputlayer(inputlayer, [00,14,14,28])
model_input_d.register_for_inputlayer(inputlayer, [14,00,28,14])

#connect inner-most layers to upper layer
model_top.register_for_ae(model_input_a)
model_top.register_for_ae(model_input_b)
model_top.register_for_ae(model_input_c)
model_top.register_for_ae(model_input_d)


inputlayer.feed_video("data/mnist.mp4", frames=25000)

model_top.transformed_summary()

model_input_a.max_activation_summary()
model_input_b.max_activation_summary()
model_input_c.max_activation_summary()
model_input_d.max_activation_summary()
model_top.max_activation_summary()

#model.save_parameters()