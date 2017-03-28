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
        with tf.device("gpu:0"):
            gan = GANNode(session,
                      name="InfoGAN",
                      loss="legacy",
                      lr=1e-3,
                      z_dim=16,
                      d_steps=1,
                      infogan=True)

            gan.register_tensor(inputlayer.get_tensor_for_region([0, 0, 32, 32]))

            gan.initialize_graph()

        self.nodes = [gan]
        self.train_op = [gan.train_op]
