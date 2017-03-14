# -*- coding: utf-8 -*-

import abc
import random
import tensorflow as tf
import numpy as np

from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli


class RegularizedGANNode(object):
    __metaclass__ = abc.ABCMeta

    # Initialization
    def __init__(self,
                 session,
                 name="gan",
                 latent_spec=[
                     (Uniform(62), False),
                     (Categorical(10), True),
                     (Uniform(1, fix_std=True), True),
                     (Uniform(1, fix_std=True), True)
                 ],
                 info_reg_coeff=1.0,
                 generator_learning_rate=1e-3,
                 discriminator_learning_rate=2e-4):

        self.name = name

        if self.name == "gan":
            self.name = 'gan_%08x' % random.getrandbits(32)

        self.session = session

        # this list is populated with register tensor function
        self.input_tensors = []
        # these are initialized upon first call to output_tensor
        self.output_tensor = None
        self.train_op = None

        # set parameters (Move those to init function params?)
        self.iteration = 0
        self.input_dim = -1
        self.latent_spec = latent_spec
        self.info_reg_coeff = info_reg_coeff
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        # generate reusable scope
        with tf.name_scope(self.name) as scope:
            self.scope = scope

        return

    def get_output_tensor(self):
        if self.output_tensor is None:
            self.initialize_graph()

        return self.output_tensor

    def initialize_graph(self):

        with tf.name_scope(self.scope):
            with tf.variable_scope(self.name):
                # concatenate input tensors
                input_concat = tf.concat(1, self.input_tensors)
                input_dim = input_concat.get_shape()[1].value
                image_shape = (int(np.sqrt(input_dim)), int(np.sqrt(input_dim)), 1)
                batch_size = input_concat.get_shape()[0].value 

                model = RegularizedGAN(
                    output_dist=MeanBernoulli(input_dim),
                    latent_spec=self.latent_spec,
                    batch_size=batch_size,
                    image_shape=image_shape,
                    network_type="mnist",
                )

            algo = InfoGANTrainer(
                name=self.name,
                model=model,
                batch_size=batch_size,
                info_reg_coeff=self.info_reg_coeff,
                generator_learning_rate=self.generator_learning_rate,
                discriminator_learning_rate=self.discriminator_learning_rate,
            )
        
            algo.input_tensor = input_concat
            algo.init_opt()
        
            self.output_tensor = model.discriminate(input_concat)[1] # TODO: which
            self.train_op = algo.generator_trainer

            ## TODO: only new variables.
            init = tf.initialize_all_variables()
            self.session.run(init)
        
    # I/O
    def register_tensor(self, new_tensor):
        # TODO: check if graph is initialized and modify for new input_dim
        self.input_tensors.append(new_tensor)
        return

    def deregister_tensor(self, tensor):
        # TODO: check if graph is initialized and modify for new input_dim
        return

    # Persistence
    def load(self, filename):
        """Retrieve model from disk."""
        return

    def save(self, filename):
        """Save model to disk."""
        return
