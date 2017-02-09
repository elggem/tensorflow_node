# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from destin.nodes import *
from destin.architectures import NetworkArchitecture

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

class DestinArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, node_type, node_params, receptive_field=[14,14], stride=[14,14]):
        """
         Todo here:

         1) create first layer nodes according to inputlayer output size, receptive field and stride
         2) append them to an array in the right order to be consumed for the second layer.. 
         3) ...do until theres only one node on top.

         - sess.run will evaluate only the needed computations, so there should be no
           redudant calcalations if we run all output_tensors, get the values, and run the training ops!
            http://stackoverflow.com/questions/34010987/does-tensorflow-rerun-for-each-eval-call
            
         - Autoencoder nodes need to publish their outputs to ROS themselves,
           after they have been evaluated.

        """

        self.nodes = []

        # Inputlayer size [28,28] : inputlayer.output_size

        debug_ae = self.create_node(session, node_type, node_params)

        debug_ae.register_tensor(inputlayer.get_tensor_for_region([0,0,28,28]))
        debug_ae.initialize_graph()

        self.nodes.append(debug_ae)
        self.train_op = debug_ae.train_op


        pass

    def create_node(self, session, node_type, node_params):
        node_class = str_to_class(node_type)
        return node_class(session, **node_params)


