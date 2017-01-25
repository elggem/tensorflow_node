#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
import tensorflow as tf
import numpy as np

from destin import AutoEncoderNode
from destin import SummaryWriter
from destin import ROSInputLayer


with tf.Session() as sess:
    rospy.init_node('destin', anonymous=False, log_level=rospy.INFO)
    
    rospy.loginfo("ROS NODE LAUNCHING")
    rospy.loginfo("recording summaries to " + SummaryWriter().get_summary_folder())
    
    ae = AutoEncoderNode(
        session=sess,
        name="ae",
        hidden_dim=40
    )

    inputlayer = ROSInputLayer(output_size=(28, 28), batch_size=250)

    ae.register_tensor(inputlayer.get_tensor_for_region([0, 14, 14, 14]))

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
    
        summary_str = merged_summary_op.eval(feed_dict=feed_dict, session=sess)
        SummaryWriter().writer.add_summary(summary_str, iteration)
        SummaryWriter().writer.flush()
    
    inputlayer.feed_topic(feed_callback, "/videofile/image_raw")
    
    rospy.spin()
