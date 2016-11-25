# -*- coding: utf-8 -*-

"""

AE Node

"""

import abc
import random
import logging as log
import tensorflow as tf
import numpy as np

from destin import SummaryWriter

class AutoEncoderNode(object):
    __metaclass__ = abc.ABCMeta

    # Initialization
    def __init__(self, 
                 session, 
                 name="ae", 
                 hidden_dim=32, 
                 activation="linear", 
                 epochs=100, 
                 noise_type="normal", 
                 noise_amount=0.2, 
                 loss="rmse", 
                 lr=0.007):

        self.name = name+'-%08x' % random.getrandbits(32)
        self.session=session

        # this list is populated 
        self.input_tensors=[]
        # these are initialized upon first call to output_tensor
        self.output_tensor=None
        self.train_op=None

        # set parameters (Move those to init function params?)
        self.iteration = 0
        self.input_dim=-1
        self.hidden_dim=hidden_dim
        self.activation=activation
        self.epochs=epochs
        self.noise_type=noise_type
        self.noise_amount=noise_amount
        self.loss=loss
        self.lr=lr

        # generate reusable scope
        with tf.name_scope(self.name) as scope:
            self.scope=scope

        return

    def get_output_tensor(self):
        if self.output_tensor == None:
            self.initialize_graph()

        return self.output_tensor

    def initialize_graph(self):
        log.debug(self.name+ " initializing output tensor...")

        # store all variables, so that we can later determinate what new variables there are
        temp = set(tf.all_variables())

        # get absolute scope
        with tf.name_scope(self.scope):
            # input placeholders            
            with tf.name_scope('input'):
                x = tf.concat(1, self.input_tensors)
                x_ = self.add_noise(x, self.noise_type, self.noise_amount)

                input_dim = x.get_shape()[1]
    
            # weight and bias variables
            with tf.variable_scope(self.name):
                encode_weights = tf.get_variable("encode_weights", (input_dim, self.hidden_dim), initializer=tf.random_normal_initializer())
                decode_weights = tf.transpose(encode_weights)
                encode_biases = tf.get_variable("encode_biases", (self.hidden_dim), initializer=tf.random_normal_initializer())
                decode_biases = tf.get_variable("decode_biases", (input_dim), initializer=tf.random_normal_initializer())

            # export weights for max activation plot
            self.encode_weights = encode_weights

            with tf.name_scope("encoded"):
                encoded = self.activate(tf.matmul(x, encode_weights) + encode_biases, self.activation, name="encoded")

            with tf.name_scope("decoded"):
                decoded = self.activate(tf.matmul(encoded, decode_weights) + decode_biases, self.activation, name="decoded")

            with tf.name_scope("loss"):
                # reconstruction loss
                if self.loss == 'rmse':
                    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
                elif self.loss == 'cross-entropy':
                    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decoded, x_))  ### TODO this is not working in it's current form! why?
                    loss = -tf.reduce_mean(x_ * tf.log(decoded))
                # record loss
            
            with tf.name_scope("train"):
                train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

            tf.scalar_summary(self.name+"_loss", loss)
            tf.histogram_summary(self.name+"_encode_weights", encode_weights)
            tf.histogram_summary(self.name+"_encode_biases",  encode_biases)
            tf.histogram_summary(self.name+"_decode_weights", decode_weights)
            tf.histogram_summary(self.name+"_decode_biases",  decode_biases)               

            # initalize all new variables
            self.session.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        
            # make train_op dependent on output tensor to trigger train on every evaluation.
            # currently train_op is triggered seperately from destin.
            
            #with self.session.graph.control_dependencies([encoded, train_op]):
            #    self.output_tensor = tf.identity(encoded, name="output_tensor")

            self.train_op = train_op
            self.output_tensor = encoded

        return


    # noise for denoising AE.
    def add_noise(self, x, noise_type, noise_amount=0.5):
        ## todo add tensorflow noise with 
        if noise_type == 'normal':
            n = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_amount, dtype=tf.float32, name="noise")
            return x + n
        elif noise_type == 'mask':
            return tf.nn.dropout(x, noise_amount, name="noise") * noise_amount
        elif noise_type == "" or noise_type == "none":
            return x


    # different activation functions
    def activate(self, linear_input, activation_type, name=None):
        if activation_type == 'sigmoid':
            return tf.nn.sigmoid(linear_input, name=name)
        elif activation_type == 'softmax':
            return tf.nn.softmax(linear_input, name=name)
        elif activation_type == 'tanh':
            return tf.nn.tanh(linear_input, name=name)
        elif activation_type == 'relu':
            return tf.nn.relu(linear_input, name=name)
        elif activation_type == 'linear':
            return linear_input

    # visualizations

    # visualization of maximum activation for all hidden neurons
    # according to: http://deeplearning.stanford.edu/wiki/index.php/Visualizing_a_Trained_Autoencoder)
    def max_activations(self):        
        outputs = []

        W = self.encode_weights.eval()

        #calculate for each hidden neuron
        for i in xrange(W.shape[1]):
            output = np.array(np.zeros(W.shape[0]),dtype='float32')
        
            W_ij_sum = 0

            for j in xrange(W.shape[0]):
                W_ij_sum += np.power(W[j][i],2)
        
            for j in xrange(W.shape[0]):
                W_ij = W[j][i]
                output[j] = (W_ij)/(np.sqrt(W_ij_sum))

            outputs.append(output)

        return outputs


    # I/O
    def register_tensor(self, new_tensor):
        #TODO: check if graph is initialized and modify for new input_dim
        self.input_tensors.append(new_tensor)
        return

    def deregister_tensor(self, tensor):
        #TODO: check if graph is initialized and modify for new input_dim
        return

    # Persistence
    def load(self, filename):
        """Retrieve model from disk."""
        return
    
    def save(self, filename):
        """Save model to disk."""
        return