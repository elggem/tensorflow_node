# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from tensorflow_node.architectures import NetworkArchitecture

class HandcodedDestinArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, node_type, node_params):
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
        self.train_op = [ae_bottom_a.train_op, ae_bottom_b.train_op, ae_bottom_c.train_op, ae_bottom_d.train_op, ae_top.train_op]


