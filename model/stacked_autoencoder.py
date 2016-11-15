# -*- coding: utf-8 -*-

import numpy as np
import random
import model.utils as utils
import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']

class StackedAutoEncoder:
    """
    A deep autoencoder with denoising capability
    based on https://github.com/rajarsheem/libsdae 
    extended for standalone use in DeSTIN perception framework
    """

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(
            self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert utils.noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse',
                 lr=0.001, batch_size=100, session=None):
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.name = "ae_%08x" % random.getrandbits(32)
        self.session = tf.Session()
        self.iteration = 0
        self.depth = len(dims)
        self.weights, self.biases = {}, {}
        self.run_operation = None
        self.saver = None

        print ("👌 Autoencoder initalized " + self.name)

    def fit(self, x):
        for i in range(self.depth):
            print(self.name + ' layer {0}'.format(i + 1)+' iteration {0}'.format(self.iteration))

            #if this is the first iteration initialize the graph
            if (self.iteration == 0):
                self.init_run(input_dim=len(x[0]),
                              hidden_dim=self.dims[i], 
                              activation=self.activations[i], 
                              loss=self.loss, 
                              lr=self.lr)

            #assert if initialization was succesful
            assert(self.run_operation and self.saver)

            if self.noise is None:
                x = self.run(data_x=x, 
                             data_x_=x,
                             epoch=self.epoch[i],
                             batch_size=self.batch_size)
            else:
                temp = np.copy(x)
                x = self.run(data_x=self.add_noise(temp),
                             data_x_=x,
                             epoch=self.epoch[i],
                             batch_size=self.batch_size)

    def transform(self, data):
        sess = self.session

        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


    def run(self, data_x, data_x_, epoch, batch_size=100):
        sess = self.session
        summary_writer = tf.train.SummaryWriter(utils.get_summary_dir(), graph=sess.graph)

        #increase iteration counter
        self.iteration = self.iteration + 1

        for i in range(epoch):
            b_x, b_x_ = utils.get_batch(data_x, data_x_, batch_size)
            _, summary_str = sess.run(self.run_operation, feed_dict={self.name+'/input/x:0': b_x, self.name+'/input/x_:0': b_x_})    
            summary_writer.add_summary(summary_str, self.iteration*epoch + i)
        
    
        #max_activations_image = utils.get_max_activation_fast(self)
        #image_summary_op = tf.image_summary("training_images", tf.reshape(max_activations_image, (1, 280, 280, 1)))
        #image_summary_str = sess.run(image_summary_op, feed_dict={x: b_x, x_: b_x_})
        #summary_writer.add_summary(image_summary_str, self.iteration)
    
        #return sess.run(encoded, feed_dict={x: data_x_})

    def init_run(self, input_dim, hidden_dim, activation, loss, lr):
        sess = self.session

        # store all variables, so that we can later determinate what new variables there are
        temp = set(tf.all_variables())

        with tf.name_scope(self.name):

            # input placeholders            
            with tf.name_scope('input'):
                x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
                x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        
            # weight and bias variables
            with tf.variable_scope(self.name) as scope:
                encode_weights = tf.get_variable("encode_weights", (input_dim, hidden_dim), initializer=tf.random_normal_initializer())
                decode_weights = tf.transpose(encode_weights)
                encode_biases = tf.get_variable("encode_biases", (hidden_dim), initializer=tf.random_normal_initializer())
                decode_biases = tf.get_variable("decode_biases", (input_dim), initializer=tf.random_normal_initializer())

            # Add summary ops to collect data
            tf.histogram_summary(self.name+"_encode_weights", encode_weights, collections=[self.name])
            tf.histogram_summary(self.name+"_encode_biases", encode_biases, collections=[self.name])
            tf.histogram_summary(self.name+"_decode_weights", decode_weights, collections=[self.name])
            tf.histogram_summary(self.name+"_decode_biases", decode_biases, collections=[self.name])

            # initialize saver for writing weights to disk
            self.saver = tf.train.Saver({"encode_weights": encode_weights, 
                                         "encode_biases": encode_biases, 
                                         "decode_biases": decode_biases})

            with tf.name_scope("encoded"):
                encoded = self.activate(tf.matmul(x, encode_weights) + encode_biases, activation)
            
            with tf.name_scope("decoded"):
                decoded = tf.matmul(encoded, decode_weights) + decode_biases
        
            with tf.name_scope("loss"):
                # reconstruction loss
                if loss == 'rmse':
                    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
                elif loss == 'cross-entropy':
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decoded, x_))  ### TODO this is not working in it's current form! why?
                # record loss
                tf.scalar_summary(self.name+"_loss", loss, collections=[self.name])
        
            with tf.name_scope("train"):
                train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    
            # Merge all summaries into a single operator
            merged_summary_op = tf.merge_all_summaries(key=self.name)
                            
            # initalize all new variables
            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

            # initalize run operation
            self.run_operation = [train_op, merged_summary_op]


    def save_weights(self):
        sess = self.session

        # Save variables to disk.
        self.saver.save(sess, utils.home_out('checkpoints')+"/"+self.name+"_"+str(self.iteration))
        print("💾 Model saved.")

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.2, (len(x), len(x[0]))).astype(x.dtype)
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), int(round(frac * len(i))), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')



