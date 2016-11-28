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
            name="bottom_a",
            hidden_dim=40

        )

    ae_bottom_b = AutoEncoderNode(
            session = sess,
            name="bottom_b",
            hidden_dim=40

        )

    ae_bottom_c = AutoEncoderNode(
            session = sess,
            name="bottom_c",
            hidden_dim=40

        )

    ae_bottom_d = AutoEncoderNode(
            session = sess,
            name="bottom_d",
            hidden_dim=40

        )

    ae_top = AutoEncoderNode(
            session = sess,
            name="top",
            hidden_dim=16
        )

    inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)
    
    ae_bottom_a.register_tensor(inputlayer.get_tensor_for_region([0,0,14,14]))
    ae_bottom_b.register_tensor(inputlayer.get_tensor_for_region([0,14,14,14]))
    ae_bottom_c.register_tensor(inputlayer.get_tensor_for_region([14,0,14,14]))
    ae_bottom_d.register_tensor(inputlayer.get_tensor_for_region([14,14,14,14]))

    ae_bottom_a.initialize_graph()
    ae_bottom_b.initialize_graph()
    ae_bottom_c.initialize_graph()
    ae_bottom_d.initialize_graph()

    ae_top.register_tensor(ae_bottom_a.get_output_tensor())
    ae_top.register_tensor(ae_bottom_c.get_output_tensor())
    ae_top.register_tensor(ae_bottom_b.get_output_tensor())
    ae_top.register_tensor(ae_bottom_d.get_output_tensor())

    ae_top.initialize_graph()

    # initialize summary writer with graph 
    SummaryWriter().writer.add_graph(sess.graph)
    merged_summary_op = tf.merge_all_summaries()

    merged_train_ops = [ae_bottom_a.train_op, ae_bottom_b.train_op, ae_bottom_c.train_op, ae_bottom_d.train_op]    

    iteration = 0

    def feed_callback(feed_dict):
        global iteration
        iteration += 1
        
        for _ in xrange(150):
            sess.run(merged_train_ops, feed_dict=feed_dict)

        for _ in xrange(150):
            sess.run(ae_top.train_op, feed_dict=feed_dict)

        #summary_str = merged_summary_op.eval(feed_dict=feed_dict)
        #SummaryWriter().writer.add_summary(summary_str, iteration)
        #SummaryWriter().writer.flush()


    inputlayer.feed_video(feed_callback, "data/mnist.mp4", frames=50000)

    image = ae_top.max_activation_recursive_summary()
    SummaryWriter().image_summary(ae_top.name, image)

    image = SummaryWriter().batch_of_1d_to_image_grid(ae_bottom_a.max_activations.eval())
    SummaryWriter().image_summary(ae_bottom_a.name, image)

    image = SummaryWriter().batch_of_1d_to_image_grid(ae_bottom_b.max_activations.eval())
    SummaryWriter().image_summary(ae_bottom_b.name, image)

    image = SummaryWriter().batch_of_1d_to_image_grid(ae_bottom_c.max_activations.eval())
    SummaryWriter().image_summary(ae_bottom_c.name, image)

    image = SummaryWriter().batch_of_1d_to_image_grid(ae_bottom_d.max_activations.eval())
    SummaryWriter().image_summary(ae_bottom_d.name, image)











