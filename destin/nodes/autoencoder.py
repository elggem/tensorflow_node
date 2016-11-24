# -*- coding: utf-8 -*-

"""

AE Node

"""

import abc
import random
import logging as log
import tensorflow as tf

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
                 noise_amount=0.5, 
                 loss="rmse", 
                 lr=0.005):

        self.name = name+'-%08x' % random.getrandbits(32)
        self.session=session
        self.input_tensors=[]
        self.output_tensor=None

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

        # do custom initialization of AE here:
        #self.output_tensor = self.init_graph()

        return

    def output(self):
        # if already initalized, return our exisiting tensor
        if self.output_tensor != None:
            return self.output_tensor

        # store all variables, so that we can later determinate what new variables there are
        temp = set(tf.all_variables())

        # get absolute scope
        with tf.name_scope(self.scope):
            # input placeholders            
            with tf.name_scope('input'):
                x = tf.concat(0, self.input_tensors)
                x = tf.reshape(x, [-1])
                x = tf.reshape(x, [1, -1]) ### TODO is this really needed, how can I get rid of batch_size?
                x_ = self.add_noise(x, self.noise_type, self.noise_amount)

                input_dim = x.get_shape()[1]
    
            # weight and bias variables
            with tf.variable_scope(self.name):
                encode_weights = tf.get_variable("encode_weights", (input_dim, self.hidden_dim), initializer=tf.random_normal_initializer())
                decode_weights = tf.transpose(encode_weights)
                encode_biases = tf.get_variable("encode_biases", (self.hidden_dim), initializer=tf.random_normal_initializer())
                decode_biases = tf.get_variable("decode_biases", (input_dim), initializer=tf.random_normal_initializer())

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

            # Add summary ops to collect data
            summary_key = self.name;

            tf.histogram_summary(self.name+"_encode_weights", encode_weights, collections=[summary_key])
            tf.histogram_summary(self.name+"_encode_biases", encode_biases, collections=[summary_key])
            tf.histogram_summary(self.name+"_decode_weights", decode_weights, collections=[summary_key])
            tf.histogram_summary(self.name+"_decode_biases", decode_biases, collections=[summary_key])
            tf.scalar_summary(self.name+"_loss", loss, collections=[summary_key])
            
            # Merge all summaries into a single operator
            merged_summary_op = tf.merge_all_summaries(key=summary_key)             

            # initalize all new variables
            self.session.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        
            with self.session.graph.control_dependencies([encoded, train_op, merged_summary_op]):
                self.output_tensor = tf.identity(encoded, name="output_tensor")

        return self.output_tensor


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


    # I/O
    def register_tensor(self, new_tensor):
        self.input_tensors.append(new_tensor)
        return

    # Persistence
    def load(self, filename):
        """Retrieve model from disk."""
        return
    
    def save(self, filename):
        """Save model to disk."""
        return