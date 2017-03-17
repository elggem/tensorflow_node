# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from tensorflow_node.architectures import NetworkArchitecture
from tensorflow_node.nodes import *

from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

class StackedInfoGANArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, latent_spec):
        
        # This is unsafe, but for research purposes I'm gonna leave this as is for now.
        latent_spec = [(Uniform(62), False)]
        for i in xrange(64):
            latent_spec.append((Uniform(1, fix_std=True), True))
            
        # Manual initialization of InfoGAN
        ##...
        infogan_a = RegularizedGANNode(
            session,
            name="gan_a",
            latent_spec=latent_spec
        )
        
        infogan_b = RegularizedGANNode(
            session,
            name="gan_b"
        )
        
        infogan_a.register_tensor(inputlayer.get_tensor_for_region([0, 0, 32, 32]))
        infogan_a.initialize_graph()
        
        infogan_b.register_tensor(infogan_a.get_output_tensor())        
        infogan_b.initialize_graph()
        
        self.nodes = [infogan_a, infogan_b]
        self.train_op = [infogan_a.train_op, infogan_b.train_op]
