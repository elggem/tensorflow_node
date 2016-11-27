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
    ae = AutoEncoderNode(
            session = sess,
            name="ae",
            hidden_dim=100

        )

    inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)
    
    ae.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))

    ae.initialize_graph()

    # initialize summary writer with graph 
    SummaryWriter().writer.add_graph(sess.graph)
    merged_summary_op = tf.merge_all_summaries()          

    iteration = 0

    def feed_callback(feed_dict):
        global iteration
        iteration += 1
        
        for _ in xrange(50):
            sess.run(ae.train_op, feed_dict=feed_dict) 

        summary_str = merged_summary_op.eval(feed_dict=feed_dict)
        SummaryWriter().writer.add_summary(summary_str, iteration)
        SummaryWriter().writer.flush()



    inputlayer.feed_video(feed_callback, "data/mnist.mp4", frames=30000)

    image = SummaryWriter().batch_of_1d_to_image_grid(ae.max_activations_tf().eval())
    SummaryWriter().image_summary(ae.name, image)
    