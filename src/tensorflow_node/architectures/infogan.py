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
        infogan_node = RegularizedGANNode(
            session,
            name="gan_node"
        )
        
        infogan_node.register_tensor(inputlayer.get_tensor_for_region([0, 0, 16, 16]))
        
        infogan_node.initialize_graph()
        
        self.nodes = [infogan_node]
        self.train_op = [infogan_node.train_op]
