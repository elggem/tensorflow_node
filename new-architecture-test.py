#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

import tensorflow as tf
import numpy as np

from destin import AutoEncoderNode
from destin import SummaryWriter
from destin import OpenCVInputLayer

log.info("recording summaries to " + SummaryWriter().get_summary_folder())

with tf.Session() as sess:
    ae_bottom_a = AutoEncoderNode(
            session = sess,
            name="bottom_a"

        )

    ae_bottom_b = AutoEncoderNode(
            session = sess,
            name="bottom_b"

        )

    ae_top = AutoEncoderNode(
            session = sess,
            name="top"
        )

    inputlayer = OpenCVInputLayer(output_size=(28,28))
    
    ae_bottom_a.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))
    ae_bottom_b.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))

    ae_top.register_tensor(ae_bottom_a.output())
    ae_top.register_tensor(ae_bottom_b.output())

    inputlayer.feed_video(ae_top.output(), "data/mnist.mp4")

    #for i in xrange(100):
    #    log.info(ae_top.output().eval(feed_dict=))


    # initialize summary writer with graph 
    SummaryWriter().writer.add_graph(sess.graph)