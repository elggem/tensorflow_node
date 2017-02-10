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

    def __init__(self, session, inputlayer, node_type, node_params, receptive_field=[14,14], stride=[7,7]):
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

        # Manual initialization of 4x4 DeSTIN

        node_params["name"] = "bottom_a"
        ae_bottom_a = self.create_node(session, node_type, node_params)

        node_params["name"] = "bottom_b"
        ae_bottom_b = self.create_node(session, node_type, node_params)
        
        node_params["name"] = "bottom_c"
        ae_bottom_c = self.create_node(session, node_type, node_params)

        node_params["name"] = "bottom_d"
        ae_bottom_d = self.create_node(session, node_type, node_params)
        
        node_params["name"] = "top"
        ae_top = self.create_node(session, node_type, node_params)

        ae_bottom_a.register_tensor(inputlayer.get_tensor_for_region([0, 0, 14, 14]))
        ae_bottom_b.register_tensor(inputlayer.get_tensor_for_region([0, 14, 14, 14]))
        ae_bottom_c.register_tensor(inputlayer.get_tensor_for_region([14, 0, 14, 14]))
        ae_bottom_d.register_tensor(inputlayer.get_tensor_for_region([14, 14, 14, 14]))

        ae_top.register_tensor(ae_bottom_a.get_output_tensor())
        ae_top.register_tensor(ae_bottom_c.get_output_tensor())
        ae_top.register_tensor(ae_bottom_b.get_output_tensor())
        ae_top.register_tensor(ae_bottom_d.get_output_tensor())

        ae_top.initialize_graph()

        self.nodes = [ae_bottom_a, ae_bottom_b, ae_bottom_c, ae_bottom_d, ae_top]
        self.train_op = [ae_bottom_a.train_op, ae_bottom_b.train_op, ae_bottom_c.train_op, ae_bottom_d.train_op, ae_top.train_op, ]


    def create_node(self, session, node_type, node_params):
        node_class = str_to_class(node_type)
        return node_class(session, **node_params)


