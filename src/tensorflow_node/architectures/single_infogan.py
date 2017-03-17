# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from tensorflow_node.architectures import NetworkArchitecture
from tensorflow_node.nodes import *

from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

class SingleInfoGANArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, latent_spec):
        
        # This is unsafe, but for research purposes I'm gonna leave this as is for now.
        latent_spec = eval(latent_spec)
        
        # Manual initialization of InfoGAN
        ##...
        infogan_node = RegularizedGANNode(
            session,
            name="gan",
            latent_spec=latent_spec
        )
        
        infogan_node.register_tensor(inputlayer.get_tensor_for_region([0, 0, 32, 32]))
        infogan_node.initialize_graph()
        
        self.nodes = [infogan_node]
        self.train_op = [infogan_node.train_op]
