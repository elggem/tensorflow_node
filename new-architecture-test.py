#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from destin import AutoEncoderNode
from destin import SummaryWriter
from destin import OpenCVInputLayer

log.info("recording summaries to " + SummaryWriter().get_summary_folder())

with tf.Session() as sess:
    ae_bottom_a = AutoEncoderNode(
            session = sess,
            name="bottom-a",
            hidden_dim=100

        )

    """
    ae_bottom_b = AutoEncoderNode(
            session = sess,
            name="bottom-b"

        )

    ae_top = AutoEncoderNode(
            session = sess,
            name="top"
        )
    """
    inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)
    
    ae_bottom_a.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))
    ae_bottom_a.initialize_graph()
    #ae_bottom_b.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))

    #ae_top.register_tensor(ae_bottom_a.get_output_tensor())
    #ae_top.register_tensor(ae_bottom_b.get_output_tensor())
    #ae_top.initialize_graph()

    # initialize summary writer with graph 
    SummaryWriter().writer.add_graph(sess.graph)
    merged_summary_op = tf.merge_all_summaries()          

    iteration = 0


    def feed_callback(feed_dict):
        global iteration
        iteration += 1
        log.info("IT %d", iteration)

        for _ in xrange(50):
            ae_bottom_a.train_op.run(feed_dict=feed_dict)

        if (iteration % 10 == 1):
            summary_str = merged_summary_op.eval(feed_dict=feed_dict)
            SummaryWriter().writer.add_summary(summary_str, iteration)
            SummaryWriter().writer.flush()

    inputlayer.feed_video(feed_callback, "data/mnist.mp4", frames=30000)

    image = SummaryWriter().batch_of_1d_to_image_grid(ae_bottom_a.max_activations())
    SummaryWriter().image_summary(ae_bottom_a.name, image)
    
    #plt.imshow(image, cmap = plt.get_cmap('gray'), interpolation='nearest')
    #plt.axis('off')
    #plt.show()
