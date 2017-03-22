# -*- coding: utf-8 -*-

"""

AE Node

"""

import abc
import random
import tensorflow as tf
import numpy as np
import rospy

from tensorflow_node import SummaryWriter


class GANNode(object):
    __metaclass__ = abc.ABCMeta

    # Initialization
    def __init__(self,
                 session,
                 name="gan",
                 loss="log"):

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
        self.loss = loss

        # generate reusable scope
        with tf.name_scope(self.name) as scope:
            self.scope = scope

        return

    def get_output_tensor(self):
        if self.output_tensor is None:
            self.initialize_graph()

        return self.output_tensor

    def initialize_graph(self):
        rospy.logdebug(self.name + " initializing output tensor...")

        # store all variables, so that we can later determinate what new variables there are
        temp = set(tf.global_variables())

        # get absolute scope
        with tf.name_scope(self.scope):
            with tf.variable_scope(self.name):

                def sample_Z(m, n):
                    '''Uniform prior for G(Z)'''
                    return tf.random_uniform([m, n], minval=-1, maxval=1)
                
                def xavier_init(size):
                    in_dim = size[0]
                    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
                    return tf.random_normal(shape=size, stddev=xavier_stddev)

                # concatenate input tensors
                input_concat = tf.concat(axis=1, values=self.input_tensors)
                input_dim = input_concat.get_shape()[1].value
                batch_size = input_concat.get_shape()[0].value

                # deep copy to prevent losses from affecting bottom layers.
                x = tf.get_variable("input_copy", input_concat.get_shape())

                # this is an operation that needs to be executed before other ops, ensure control dependency!
                assign = x.assign(input_concat)

                # Discriminator Net
                X = input_concat

                D_W1 = tf.Variable(xavier_init([input_dim, 128]), name='D_W1')
                D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

                D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
                D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

                theta_D = [D_W1, D_W2, D_b1, D_b2]

                # Generator Net
                #Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
                Z = sample_Z(batch_size, 100)
                
                G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
                G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

                G_W2 = tf.Variable(xavier_init([128, input_dim]), name='G_W2')
                G_b2 = tf.Variable(tf.zeros(shape=[input_dim]), name='G_b2')

                theta_G = [G_W1, G_W2, G_b1, G_b2]

                def generator(z):
                    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
                    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
                    G_prob = tf.nn.sigmoid(G_log_prob)

                    return G_prob


                def discriminator(x):
                    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
                    D_logit = tf.matmul(D_h1, D_W2) + D_b2
                    D_prob = tf.nn.sigmoid(D_logit)

                    return D_prob, D_logit

                with self.session.graph.control_dependencies([assign]):
                    G_sample = generator(Z)
                    D_real, D_logit_real = discriminator(X)
                    D_fake, D_logit_fake = discriminator(G_sample)

                    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
                    G_loss = -tf.reduce_mean(tf.log(D_fake))

                    # Only update D(X)'s parameters, so var_list = theta_D
                    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
                    # Only update G(X)'s parameters, so var_list = theta_G
                    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
                
                self.train_op = [D_solver, G_solver]
                self.output_tensor = D_real
                
                tf.summary.scalar(self.name + "_D_loss", D_loss)
                tf.summary.scalar(self.name + "_G_loss", G_loss)


            # initalize all new variables
            self.session.run(tf.variables_initializer(set(tf.global_variables()) - temp))

        return

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
