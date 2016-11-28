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

        # this list is populated with register tensor function
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
        log.debug(self.name+ " initializing output tensor...    ")

        # store all variables, so that we can later determinate what new variables there are
        temp = set(tf.all_variables())

        # get absolute scope
        with tf.name_scope(self.scope):
            
            with tf.variable_scope(self.name):
                # concatenate input tensors
                input_concat = tf.concat(1, self.input_tensors)
                input_dim = input_concat.get_shape()[1]

                # deep copy to prevent losses from affecting bottom layers.
                x = tf.get_variable("input_copy", input_concat.get_shape())
                x_ = self.add_noise(x, self.noise_type, self.noise_amount)

                # this is an operation that needs to be executed before other ops, ensure control dependency!
                assign = x.assign(input_concat)

                encode_weights = tf.get_variable("encode_weights", (input_dim, self.hidden_dim), initializer=tf.random_normal_initializer())
                decode_weights = tf.transpose(encode_weights)
                encode_biases = tf.get_variable("encode_biases", (self.hidden_dim), initializer=tf.random_normal_initializer())
                decode_biases = tf.get_variable("decode_biases", (input_dim), initializer=tf.random_normal_initializer())

            # visualization of maximum activation for all hidden neurons
            # according to: http://deeplearning.stanford.edu/wiki/index.php/Visualizing_a_Trained_Autoencoder
            with tf.name_scope("max_activations"):
                self.max_activations = tf.transpose(encode_weights / tf.reduce_sum(tf.pow(encode_weights, 2)))

            # ensure deep copy for these operations
            with self.session.graph.control_dependencies([assign]):
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

            # attach reference to ourselve for recursive plot.
            train_op.sender = self
            encoded.sender = self

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

    def max_activation_recursive(self):
        ## refactor make this work without loops.
        recursive_activations = []

        i = 0
        #1 calculate max_activation (hidden x input_dim matrix)
        for max_activation in self.max_activations.eval():
            #log.critical("looking at hidden neuron " + str(i))
            i += 1
            #2 for each input_dim matrix split it up according to input_buffer (AE: sender.ndims[-1] - inputlayer: sender.dims_for_receiver(self))
            dimcounter = 0
            activation = []

            for input_tensor in self.input_tensors:
                sender = input_tensor.sender
                ndims = input_tensor.get_shape()[1].value
                #log.critical("   looking at " + sender.name)
                if (sender.__class__ == self.__class__):
                    sender_activation = max_activation[dimcounter:dimcounter+ndims]
                    log.critical("Got slice " + str(dimcounter) +" to " + str(dimcounter+ndims) + " from max_activation.")
                    dimcounter += ndims
                    #3 for each input_buffer that is AE ask for max_activation object and multiply
                    sender_max_activations = sender.max_activation_recursive()
                    #sender activation = |hl| and sender_max_activations = hl x input
                    A = np.array(sender_activation)
                    B = np.array(sender_max_activations)
                    C = (A[:, np.newaxis] * B).sum(axis=0)
                    activation.append(C)

                elif (str(sender.__class__).find("InputLayer")):
                    sender_activation = max_activation[dimcounter:dimcounter+ndims]
                    dimcounter += ndims
                    #4 for each input_buffer that is input_layer return it
                    activation.append(sender_activation)
            
            recursive_activations.append(np.concatenate(activation))

        recursive_activations = np.array(recursive_activations)
        print recursive_activations.shape

        return recursive_activations

   
    def max_activation_recursive_summary(self):
        # TODO: this is horrible, but it works. :)
        sess = self.session

        outputs = np.array(self.max_activation_recursive()) ## needs to be reshaped.
        shaped_outputs = []

        input_wh = int(np.ceil(np.power(outputs.shape[1],0.5)))
        input_shape = [input_wh, input_wh]

        for output in outputs:
            #output 0-40, 40-80, 80-120, 120-160
            A = np.concatenate([output[:196].reshape([14,14]), output[196:392].reshape([14,14])], axis=0) ### TODO: this is hardcoded.
            B = np.concatenate([output[392:588].reshape([14,14]), output[588:784].reshape([14,14])], axis=0)
            shaped_outputs.append(np.concatenate([A,B], axis=1))

        output_wh = int(np.floor(np.power(outputs.shape[0],0.5)))
        output_shape = [input_wh*output_wh, input_wh*output_wh]
        output_rows = []
        
        activation_image = np.zeros(output_shape, dtype=np.float32)

        print output_shape

        for i in xrange(output_wh):
            output_rows.append(np.concatenate(shaped_outputs[i*output_wh:(i*output_wh)+output_wh], 0))

        activation_image = np.concatenate(output_rows, 1)

        return activation_image




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