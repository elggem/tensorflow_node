# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from tensorflow_node.architectures import NetworkArchitecture
from tensorflow_node.nodes import *

class SingleGANArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, foo):
        # Manual initialization of 4x4 DeSTIN

        gan = GANNode(session)

        gan.register_tensor(inputlayer.get_tensor_for_region([0, 0, 14, 14]))

        gan.initialize_graph()

        self.nodes = [gan]
        self.train_op = [gan.train_op]
