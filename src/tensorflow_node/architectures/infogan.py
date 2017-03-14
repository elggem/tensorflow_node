# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from tensorflow_node.architectures import NetworkArchitecture
from tensorflow_node.nodes import *

class InfoGANArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer):
        # Manual initialization of InfoGAN
        ##...
        infogan_a = RegularizedGANNode(
            session,
            name="gan_a"
        )
        
        infogan_b = RegularizedGANNode(
            session,
            name="gan_b"
        )
        
        infogan_a.register_tensor(inputlayer.get_tensor_for_region([0, 0, 8, 8]))
        infogan_b.register_tensor(inputlayer.get_tensor_for_region([8, 8, 8, 8]))
        
        infogan_a.initialize_graph()
        infogan_b.initialize_graph()
        
        self.nodes = [infogan_a, infogan_b]
        self.train_op = [infogan_a.train_op, infogan_b.train_op]
