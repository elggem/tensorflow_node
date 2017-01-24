#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
import tensorflow as tf
import numpy as np

from destin import AutoEncoderNode
from destin import SummaryWriter
from destin import ROSInputLayer

rospy.init_node('destin', anonymous=False, log_level=rospy.INFO)

rospy.loginfo("ROS NODE LAUNCHING")
rospy.loginfo("recording summaries to " + SummaryWriter().get_summary_folder())

inputlayer = ROSInputLayer(output_size=(28, 28), batch_size=250)


def feed_callback(feed_dict):
    global iteration
    iteration += 1

    for _ in xrange(50):
        sess.run(merged_train_ops, feed_dict=feed_dict)
        sess.run(ae_top.train_op, feed_dict=feed_dict)

    summary_str = merged_summary_op.eval(feed_dict=feed_dict)
    SummaryWriter().writer.add_summary(summary_str, iteration)
    SummaryWriter().writer.flush()

inputlayer.feed_topic(feed_callback, "/videofile/image_raw")

rospy.spin()
