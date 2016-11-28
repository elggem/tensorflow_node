# -*- coding: utf-8 -*-

import abc
import random
import logging as log
import tensorflow as tf

from .autoencoder import AutoEncoderNode


class StackedAutoEncoderNode(object):
    __metaclass__ = abc.ABCMeta

    # Initialization
    def __init__(self,
                 session,
                 name="sae",
                 hidden_dims=[32, 32],
                 activations=["linear", "linear"],
                 noise_type="normal",
                 noise_amount=0.2,
                 loss="rmse",
                 lr=0.007):

        self.name = name + '-%08x' % random.getrandbits(32)
        self.session = session

        # this list is populated with register tensor function
        self.input_tensors = []
        # these are initialized upon first call to output_tensor
        self.output_tensor = None
        self.train_op = None

        # set parameters (Move those to init function params?)
        self.iteration = 0
        self.input_dim = -1
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.noise_type = noise_type
        self.noise_amount = noise_amount
        self.loss = loss
        self.lr = lr

        self.autoencoders = []

        for hidden_dim, activation in zip(self.hidden_dims, self.activations):
            ae = AutoEncoderNode(
                session=self.session,
                name="ae",
                hidden_dim=hidden_dim,
                activation=activation,
                noise_type=self.noise_type,
                noise_amount=self.noise_amount,
                loss=self.loss,
                lr=self.lr
            )

            self.autoencoders.append(ae)

        return

    def get_output_tensor(self):
        if self.output_tensor is None:
            self.initialize_graph()

        return self.output_tensor

    def initialize_graph(self):
        self.train_op = []

        for i, ae in enumerate(self.autoencoders):
            if (i > 0):
                ae.register_tensor(self.autoencoders[i - 1].get_output_tensor())

            if (i == len(self.autoencoders) - 1):
                self.output_tensor = ae.get_output_tensor()

        for ae in self.autoencoders:
            self.train_op.append(ae.train_op)

    # I/O
    def register_tensor(self, new_tensor):
        # TODO: check if graph is initialized and modify for new input_dim
        self.autoencoders[0].register_tensor(new_tensor)
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
