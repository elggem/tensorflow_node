#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import rospy
import tensorflow as tf
import numpy as np

from destin import *

from std_msgs.msg import Header
from ros_destin.msg import DestinNodeState

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

with tf.Session() as sess:
    # initialize ROS node
    rospy.init_node('destin', anonymous=False, log_level=rospy.INFO)
    rospy.loginfo("Destin ROS node launching")

    # initialize input layer from yaml
    inputlayer_type = rospy.get_param("inputlayer/type")
    inputlayer_class = str_to_class(inputlayer_type)
    inputlayer_params = rospy.get_param("inputlayer/params")
    inputlayer = inputlayer_class(**inputlayer_params)

    # initialize network from yaml
    architecture_type = rospy.get_param("architecture/type")
    architecture_class = str_to_class(architecture_type)
    architecture_params = rospy.get_param("architecture/params")
    architecture = architecture_class(sess, inputlayer, **architecture_params)

    # initialize summary writer
    if (rospy.get_param("publishing/summaries")):
        rospy.loginfo("recording summaries to " + SummaryWriter().get_summary_folder())
        # initialize summary writer with graph
        SummaryWriter().writer.add_graph(sess.graph)
        merged_summary_op = tf.merge_all_summaries()

    # initialize publishers for network
    publishers = {}
    topic_name = rospy.get_param("publishing/topic")
    for node in architecture.nodes:
        publishers[node.name] = rospy.Publisher('/'+topic_name+'/'+node.name, DestinNodeState, queue_size=rospy.get_param("inputlayer")['batch_size'])

    # main callback to evaluate architecture and publish states
    iteration = 0

    def feed_callback(feed_dict):
        global iteration
        iteration += 1

        # TODO: Is it necessary to group op's here better?

        # Execute train_op for entire network architecture
        for _ in xrange(50): # TODO parametrize this
            sess.run(architecture.train_op, feed_dict=feed_dict)

        # iterate over each state and stream output to ROS
        state_ops = []
        for node in architecture.nodes:
            ae_state = sess.run(node.get_output_tensor(), feed_dict=feed_dict)

            for state in ae_state:
                # formulate message
                msg = DestinNodeState()
    
                msg.header = Header()
                msg.header.stamp = rospy.Time.now()
                msg.id = node.name
                msg.type = node.__class__.__name__
                msg.state = state
                # TODO input_nodes, output_nodes    

                # publish message
                publishers[node.name].publish(msg)
        
        # publish summary output
        if (rospy.get_param("publishing/summaries")):
            summary_str = merged_summary_op.eval(feed_dict=feed_dict, session=sess)
            SummaryWriter().writer.add_summary(summary_str, iteration)
            SummaryWriter().writer.flush()

        # quit gracefully
        if (rospy.is_shutdown()):
            # TODO: checkpoint model here
            print("\nExiting DeSTIN ✌️ ")
            sys.exit(0)
    
    # start feeding in data to callback
    inputlayer.feed_to(feed_callback)    

    # rospy spin
    rospy.spin()



