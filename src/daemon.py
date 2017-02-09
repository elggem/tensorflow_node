#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import rospy
import tensorflow as tf
import numpy as np

from destin import *

from std_msgs.msg import Header
from ros_destin.msg import DestinNodeState

# TODO: move this to utils
def str_to_class(str):
    return getattr(sys.modules[__name__], str)

# This is how to get parameters:

# ###
# 1) XX Iniitalize input layer from YAML configuration (default?)
# 2) Initialize network architecture using class and params specified in YAML
# 3) Connect everything up 
#
# 4) Whenever there is new frame on input layer (if it's ros, otherwise itll just run through and publish)
#
# - publish state from NODE instead of here!
# ###


with tf.Session() as sess:
    rospy.init_node('destin', anonymous=False, log_level=rospy.INFO)
    
    rospy.loginfo("ROS NODE LAUNCHING")
    rospy.loginfo("recording summaries to " + SummaryWriter().get_summary_folder())
    
    ae = AutoEncoderNode(
        session=sess,
        name="ae",
        hidden_dim=40
    )

    # initialize input layer from yaml
    inputlayer = str_to_class(rospy.get_param("inputlayer/type"))(**rospy.get_param("inputlayer"))

    #inputlayer = ROSInputLayer(output_size=(28, 28), batch_size=250)

    ae.register_tensor(inputlayer.get_tensor_for_region([0, 14, 14, 14]))

    ae.initialize_graph()

    # initialize summary writer with graph
    SummaryWriter().writer.add_graph(sess.graph)
    merged_summary_op = tf.merge_all_summaries()

    iteration = 0

    pub = rospy.Publisher('/destin/'+ae.name, DestinNodeState, queue_size=rospy.get_param("inputlayer")['batch_size'])
    #pub.publish(std_msgs.msg.String("foo"))    

    def feed_callback(feed_dict):
        global iteration
        iteration += 1


        for _ in xrange(50):
            sess.run(ae.train_op, feed_dict=feed_dict)

        ae_state = ae.output_tensor.eval(feed_dict=feed_dict, session=sess)

        for state in ae_state:
            ## publish state
            msg = DestinNodeState()

            msg.header = Header()
            msg.header.stamp = rospy.Time.now() # Note you need to call rospy.init_node() before this will work

            msg.id = ae.name
            msg.type = ae.__class__.__name__
            ##todo input_nodes, output_nodes

            # TODO Here we only take the state of the first input...
            msg.state = state

            pub.publish(msg)
    
        summary_str = merged_summary_op.eval(feed_dict=feed_dict, session=sess)
        SummaryWriter().writer.add_summary(summary_str, iteration)
        SummaryWriter().writer.flush()
    
    inputlayer.feed_to(feed_callback)
    #inputlayer.feed_topic(feed_callback, "/videofile/image_raw")
    
    rospy.spin()

