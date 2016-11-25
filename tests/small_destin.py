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

    inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=250)
    
    ae_bottom_a.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))
    ae_bottom_b.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))

    ae_top.register_tensor(ae_bottom_a.get_output_tensor())
    ae_top.register_tensor(ae_bottom_b.get_output_tensor())
    ae_top.initialize_graph()

    merged_summary_op = tf.merge_all_summaries()          

    iteration = 0

    def feed_callback(feed_dict):
        global iteration
        iteration += 1
        
        #sess.run([ae_bottom_a.train_op, ae_bottom_b.train_op, ae_top.train_op], feed_dict=feed_dict)
        sess.run(ae_bottom_a.train_op, feed_dict=feed_dict)

        #summary_str = merged_summary_op.eval(feed_dict=feed_dict)
        #SummaryWriter().writer.add_summary(summary_str, iteration)
        #SummaryWriter().writer.flush()


        



    inputlayer.feed_video(feed_callback, "data/mnist.mp4", frames=30000)


    # initialize summary writer with graph 
    SummaryWriter().writer.add_graph(sess.graph)