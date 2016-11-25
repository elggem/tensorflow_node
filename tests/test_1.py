#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline

from destin import AutoEncoderNode
from destin import SummaryWriter
from destin import OpenCVInputLayer

log.info("recording summaries to " + SummaryWriter().get_summary_folder())

with tf.Session() as sess:
    inputlayer = OpenCVInputLayer(output_size=(28,28), batch_size=100)

    ae_bottom_a = AutoEncoderNode(
            session = sess,
            name="bottom-a",
            hidden_dim=100

        )

    ae_bottom_b = AutoEncoderNode(
            session = sess,
            name="bottom-b",
            hidden_dim=100

        )

    ae_bottom_a.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))
    ae_bottom_a.initialize_graph()

    ae_bottom_b.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))
    ae_bottom_b.initialize_graph()

    # initialize summary writer with graph 
    SummaryWriter().writer.add_graph(sess.graph)
    merged_summary_op = tf.merge_all_summaries()          
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()

    iteration = 0

    def feed_callback(feed_dict):
        global iteration
        iteration += 1
        log.info("IT %d", iteration)

        #for _ in xrange(50):
        summary_str, _,_ = sess.run([merged_summary_op, ae_bottom_a.train_op,ae_bottom_b.train_op], feed_dict=feed_dict)

        SummaryWriter().writer.add_summary(summary_str, iteration)
        SummaryWriter().writer.flush()


    inputlayer.feed_video(feed_callback, "data/mnist.mp4", frames=10000)

    #SummaryWriter().writer.add_run_metadata(run_metadata, "run")

    #tl = timeline.Timeline(run_metadata.step_stats)
    #ctf = tl.generate_chrome_trace_format()
    #with open(SummaryWriter().get_output_folder('timelines')+"/timeline.json", 'w') as f:
    #    f.write(ctf)
    #    log.info("📊 written timeline trace.")

    #image = SummaryWriter().batch_of_1d_to_image_grid(ae_bottom_a.max_activations())
    #SummaryWriter().image_summary(ae_bottom_a.name, image)
    
    #plt.imshow(image, cmap = plt.get_cmap('gray'), interpolation='nearest')
    #plt.axis('off')
    #plt.show()
