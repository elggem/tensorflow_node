# -*- coding: utf-8 -*-

"""

GAN Node

heaviliy influenced by https://github.com/wiseodd/generative-models

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
                 h_dim=128,
                 z_dim=64,
                 d_steps=3,
                 name="gan",
                 loss="log",
                 infogan=False,
                 lr = 1e-3):

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
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.d_steps = d_steps
        self.loss = loss
        self.lr = lr
        self.infogan = infogan
        
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
                    return tf.random_uniform([m, n], minval=-1, maxval=1) # TODO: Use Normal dist here.
                
                # for InfoGAN
                def sample_c(m):
                    #return np.random.multinomial(1, 10*[0.1], size=m)
                    onehot = tf.one_hot(tf.multinomial(tf.log([[10.,10.]*5]),m),10)
                    return tf.reshape(onehot, [m,-1])

                
                def xavier_init(size):
                    in_dim = size[0]
                    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
                    return tf.random_normal(shape=size, stddev=xavier_stddev)

                # concatenate input tensors
                input_concat = tf.concat(axis=1, values=self.input_tensors)
                input_dim = input_concat.get_shape()[1].value
                batch_size = input_concat.get_shape()[0].value
                image_shape = [int(np.sqrt(input_dim)),int(np.sqrt(input_dim))]

                # deep copy to prevent losses from affecting bottom layers.
                X = tf.get_variable("input_copy", input_concat.get_shape())

                # this is an operation that needs to be executed before other ops, ensure control dependency!
                assign = X.assign(input_concat)

                # Discriminator Net
                D_W1 = tf.Variable(xavier_init([input_dim, self.h_dim]), name='D_W1')
                D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]), name='D_b1')

                D_W2 = tf.Variable(xavier_init([self.h_dim, 1]), name='D_W2')
                D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

                theta_D = [D_W1, D_W2, D_b1, D_b2]

                # Generator Net
                Z = sample_Z(batch_size, self.z_dim)
                
                if self.infogan:
                    self.z_dim += 10
                
                G_W1 = tf.Variable(xavier_init([self.z_dim, self.h_dim]), name='G_W1')
                G_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]), name='G_b1')

                G_W2 = tf.Variable(xavier_init([self.h_dim, input_dim]), name='G_W2')
                G_b2 = tf.Variable(tf.zeros(shape=[input_dim]), name='G_b2')

                theta_G = [G_W1, G_W2, G_b1, G_b2]

                ## InfoGAN
                if self.infogan:
                    Q_W1 = tf.Variable(xavier_init([input_dim, self.h_dim]))
                    Q_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

                    Q_W2 = tf.Variable(xavier_init([self.h_dim, 10]))
                    Q_b2 = tf.Variable(tf.zeros(shape=[10]))

                    theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]

                    def Q(x):
                        Q_h1 = tf.nn.relu(tf.matmul(x, Q_W1) + Q_b1)
                        Q_prob = tf.nn.softmax(tf.matmul(Q_h1, Q_W2) + Q_b2)
                        return Q_prob

                def generator(z, c=None):
                    if c != None:
                        inputs = tf.concat(axis=1, values=[z, c])
                    else:
                        inputs = z

                    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
                    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
                    G_prob = tf.nn.sigmoid(G_log_prob)

                    return G_prob

                def discriminator(x):
                    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
                    D_logit = tf.matmul(D_h1, D_W2) + D_b2
                    D_prob = tf.nn.sigmoid(D_logit)

                    return D_prob, D_logit

                with self.session.graph.control_dependencies([assign]):

                    # InfoGAN
                    if self.infogan:
                        c = sample_c(batch_size)
                        G_sample = generator(Z, c)
                        Q_c_given_x = Q(G_sample)
                        
                        latent_variables = Q(X)
                        
                        cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_given_x + 1e-8) * c, 1))
                        ent = tf.reduce_mean(-tf.reduce_sum(tf.log(c + 1e-8) * c, 1))
                        Q_loss = cond_ent + ent
                    else:
                        G_sample = generator(Z)
                    
                    D_real, D_logit_real = discriminator(X)
                    D_fake, D_logit_fake = discriminator(G_sample)

                    if self.loss == "log":
                        # alternative loss
                        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
                        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
                        D_loss = D_loss_real + D_loss_fake
                        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))    
                    elif self.loss == "ls":
                        # LSGAN 
                        D_loss = 0.5 * (tf.reduce_mean((D_logit_real - 1)**2) + tf.reduce_mean(D_logit_fake**2))
                        G_loss = 0.5 * tf.reduce_mean((D_logit_fake - 1)**2)                        
                    elif self.loss == "wasserstein":
                        # Wasserstein-GAN
                        D_loss = tf.reduce_mean(D_logit_real) - tf.reduce_mean(D_logit_fake)
                        G_loss = tf.reduce_mean(D_logit_fake)
                        clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]
                    elif self.loss == "legacy":
                        D_loss = -tf.reduce_mean(tf.log(D_real + 1e-8) + tf.log(1 - D_fake + 1e-8))
                        G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-8))
                    else:
                        raise NotImplementedError

                    # Only update D(X)'s parameters, so var_list = theta_D
                    D_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(D_loss, var_list=theta_D)
                    with self.session.graph.control_dependencies([D_solver]):
                        # Only update G(X)'s parameters, so var_list = theta_G
                        G_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(G_loss, var_list=theta_G)

                    self.output_tensor = D_real#latent_variables
                    self.train_op = []
                    
                    for _ in range(self.d_steps):
                        self.train_op.append(D_solver)

                    if self.loss == "wasserstein":
                        self.train_op.insert(1,clip_D)
                
                    self.train_op.append(G_solver)
                    
                    # InfoGAN
                    if self.infogan:
                        with self.session.graph.control_dependencies([G_solver]):
                            Q_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(Q_loss, var_list=theta_G + theta_Q)
                        self.train_op.append(Q_solver)
                    
                    # Summaries
                    with tf.device("/cpu:0"):
                        tf.summary.scalar(self.name + "_D_loss", D_loss)
                        tf.summary.scalar(self.name + "_G_loss", G_loss)
                        tf.summary.histogram(self.name + "_D_real", D_real)
                        tf.summary.histogram(self.name + "_D_fake", D_fake)

                        tf.summary.image(self.name + "real_sample", tf.reshape(X, [-1,image_shape[0],image_shape[1],1]), max_outputs=5)
                        tf.summary.image(self.name + "fake_sample", tf.reshape(G_sample, [-1,image_shape[0],image_shape[1],1]), max_outputs=5)
                        
                        if self.infogan:
                            tf.summary.scalar(self.name + "_Q_loss", Q_loss)  
                            tf.summary.histogram(self.name + "_latent_variables", latent_variables)
                            
                            # Generate latent var pictures for summary #todo: tiles.
                            for i in xrange(10):
                                G_label = generator(sample_Z(5, self.z_dim-10), tf.one_hot([i]*5,10))
                                tf.summary.image(self.name + "infogan_sample_%d" % i, tf.reshape(G_label, [-1,image_shape[0],image_shape[1],1]), max_outputs=5)
                            else:
                                G_label = generator(sample_Z(5, self.z_dim))
                                tf.summary.image(self.name + "gan_sample", tf.reshape(G_label, [-1,image_shape[0],image_shape[1],1]), max_outputs=5)

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
