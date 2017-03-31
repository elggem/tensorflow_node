# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from tensorflow_node.architectures import NetworkArchitecture
from tensorflow_node.nodes import *

class SingleGANArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, foo):

        # Debug for MNIST
        latent_vars = [("categorical", 10)]#, ("uniform", True)]
        #latent_vars = [("uniform", True), ("uniform", True), ("uniform", True), ("uniform", True), ("uniform", True)]
        #latent_vars = [("categorical", 10), ("categorical", 2)]#, ("uniform", 1), ("uniform", 1)]
        
        #latent_vars = None

        with tf.device("gpu:0"):
            gan = GANNode(session,
                      name="InfoGAN",
                      loss="legacy",
                      lr=1e-3,
                      z_dim=16,
                      d_steps=1,
                      latent_vars=latent_vars)

            gan.register_tensor(inputlayer.get_tensor_for_region([0, 0, 32, 32]))

            gan.initialize_graph()

        self.nodes = [gan]
        self.train_op = [gan.train_op]
