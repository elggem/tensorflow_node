# -*- coding: utf-8 -*-

import numpy as np
import random
import model.utils as utils
from summary_writer import SummaryWriter

import tensorflow as tf
from tensorflow.python.client import timeline
allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']

class StackedAutoEncoder:
    """
    A deep autoencoder with denoising capability
    based on https://github.com/rajarsheem/libsdae 
    extended for standalone use in DeSTIN perception framework
    """

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse', lr=0.001, session=None, metadata=False, timeline=False):
        # object initialization
        self.name = "ae-%08x" % random.getrandbits(32)
        self.session = tf.Session()
        self.iteration = 0 # one iteration is fitting batch_size*epoch times.
        # parameters
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.metadata = metadata # collect metadata information
        self.timeline = timeline # collect timeline information
        self.assertions()

        # layer variables and ops
        self.depth = len(dims)
        self.layers = []
        for i in xrange(self.depth):
            self.layers.append({'encode_weights': None, 
                                'encode_biases': None, 
                                'decode_biases': None, 
                                'run_op': None, 
                                'summ_op': None, 
                                'encode_op': None, 
                                'decode_op': None})

        # callback to other autoencoders
        self.callback = None ##called when result is available.

        # namescope for summary writers
        with tf.name_scope(self.name) as scope:
            self.scope = scope

        print ("👌 Autoencoder initalized " + self.name)

    def __del__(self):
        self.session.close()
        print ("🖐 Autoencoder " + self.name + " deallocated, closed session.")


    def fit(self, x):
        # increase iteration counter
        self.iteration += 1

        for i in range(self.depth):
            print(self.name + ' layer {0}'.format(i + 1)+' iteration {0}'.format(self.iteration))

            #if this is the first iteration initialize the graph
            if (self.iteration == 1):
                self.init_run(input_dim=len(x[0]),
                              layer=i,
                              hidden_dim=self.dims[i], 
                              activation=self.activations[i], 
                              loss=self.loss, 
                              lr=self.lr)

            if self.noise is None:
                x = self.run(data_x=x, 
                             data_x_=x,
                             layer=i,
                             epoch=self.epoch[i])
            else:
                temp = np.copy(x)
                x = self.run(data_x=self.add_noise(temp),
                             data_x_=x,
                             layer=i,
                             epoch=self.epoch[i])


    def transform(self, data):
        sess = self.session

        x = tf.constant(data, dtype=tf.float32)
        for layer, a in zip(self.layers, self.activations):
            layer = tf.matmul(x, layer['encode_weights']) + layer['encode_biases']
            x = self.activate(layer, a)
        transformed = x.eval(session=sess)
        return transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def run(self, data_x, data_x_, layer, epoch):
        sess = self.session

        if self.metadata:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        feeding_scope = self.name+"/layer_"+str(layer)+"/input/"

        # fit the model according to number of epochs:
        for i in range(epoch):
            feed_dict = {feeding_scope+'x:0':  data_x, feeding_scope+'x_:0': data_x_}
            sess.run(self.layers[layer]['run_op'], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            

        # create the timeline object, and write it to a json
        if self.metadata and self.timeline:
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(utils.home_out('timelines')+"/"+self.name+"_layer_"+str(layer)+"_iteration_"+str(self.iteration)+".json", 'w') as f:
                f.write(ctf)
                print "📊 written timeline trace."

        # put metadata into summaries.
        if self.metadata:
            SummaryWriter().writer.add_run_metadata(run_metadata, 'step%d' % (self.iteration*epoch + i))

        # run summary operation.
        summary_str = sess.run(self.layers[layer]['summ_op'], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        SummaryWriter().writer.add_summary(summary_str, self.iteration*epoch + i)
        SummaryWriter().writer.flush()

        return sess.run(self.layers[layer]['encode_op'], feed_dict={feeding_scope+'x:0': data_x_})


    def init_run(self, input_dim, hidden_dim, activation, loss, lr, layer):
        sess = self.session

        # store all variables, so that we can later determinate what new variables there are
        temp = set(tf.all_variables())

        # get absolute scope
        with tf.name_scope(self.scope):
            with tf.name_scope("layer_"+str(layer)):
                # input placeholders            
                with tf.name_scope('input'):
                    x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
                    x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
    
                # weight and bias variables
                with tf.variable_scope(self.name):
                    with tf.variable_scope("layer_"+str(layer)):
                        encode_weights = tf.get_variable("encode_weights", (input_dim, hidden_dim), initializer=tf.random_normal_initializer())
                        decode_weights = tf.transpose(encode_weights) ## AE is symmetric (bound variables), thus no seperate decoder weights.
                        encode_biases = tf.get_variable("encode_biases", (hidden_dim), initializer=tf.random_normal_initializer())
                        decode_biases = tf.get_variable("decode_biases", (input_dim), initializer=tf.random_normal_initializer())

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
                
                with tf.name_scope("train"):
                    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

                # Add summary ops to collect data
                summary_key = self.name+"_layer_"+str(layer);

                tf.histogram_summary(self.name+"_encode_weights_layer_"+str(layer), encode_weights, collections=[summary_key])
                tf.histogram_summary(self.name+"_encode_biases_layer_"+str(layer), encode_biases, collections=[summary_key])
                tf.histogram_summary(self.name+"_decode_weights_layer_"+str(layer), decode_weights, collections=[summary_key])
                tf.histogram_summary(self.name+"_decode_biases_layer_"+str(layer), decode_biases, collections=[summary_key])
                tf.scalar_summary(self.name+"_loss_layer_"+str(layer), loss, collections=[summary_key])
                
                # Merge all summaries into a single operator
                merged_summary_op = tf.merge_all_summaries(key=summary_key)             
                
                # initialize summary writer with graph 
                SummaryWriter().writer.add_graph(sess.graph)

                # initalize all new variables
                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        
                # initalize accessor variables
                self.layers[layer]['encode_weights'] = encode_weights
                self.layers[layer]['encode_biases'] = encode_biases
                self.layers[layer]['decode_biases'] = decode_biases

                self.layers[layer]['encode_op'] = encoded
                self.layers[layer]['decode_op'] = decoded
                
                self.layers[layer]['run_op'] = train_op
                self.layers[layer]['summ_op'] = merged_summary_op


    def save_parameters(self):
        sess = self.session

        to_be_saved = {}

        for layer in xrange(self.depth):
            to_be_saved['layer'+str(layer)+'_encode_weights'] = self.layers[layer]['encode_weights']
            to_be_saved['layer'+str(layer)+'_encode_biases'] = self.layers[layer]['encode_biases']
            to_be_saved['layer'+str(layer)+'_decode_biases'] = self.layers[layer]['decode_biases']

        saver = tf.train.Saver(to_be_saved)
        saver.save(sess, utils.home_out('checkpoints')+"/"+self.name+"_"+str(self.iteration))

        print("💾 model saved.")

    def load_parameters(self, filename):
        raise NotImplementedError()
        #TODO
        #self.saver.restore(sess, ....)
        #print("💾✅ model restored.")


    def write_activation_summary(self):
        sess = self.session

        #layer 0 not initialized.
        if (not self.layers[0]['encode_weights'] == None):
            pass

        W = self.layers[0]['encode_weights'].eval(session=sess)

        input_wh = int(np.ceil(np.power(W.shape[0],0.5)))
        input_shape = [input_wh, input_wh]

        output_wh = int(np.ceil(np.power(W.shape[1],0.5)))
        output_shape = [input_wh*output_wh, input_wh*output_wh]

        outputs = []
        output_rows = []

        activation_image = np.zeros(output_shape, dtype=np.float32)

        #calculate for each hidden neuron
        for i in xrange(W.shape[1]):
            output = np.array(np.zeros(W.shape[0]),dtype='float32')
        
            W_ij_sum = 0

            for j in xrange(output.size):
                W_ij_sum += np.power(W[j][i],2)
        
            for j in xrange(output.size): 
                W_ij = W[j][i]
                output[j] = (W_ij)/(np.sqrt(W_ij_sum))
        
            outputs.append(output.reshape(input_shape))
        
        for i in xrange(10):
            output_rows.append(np.concatenate(outputs[i*10:(i*10)+10], 0))

        activation_image = np.concatenate(output_rows, 1)
        
        image_summary_op = tf.image_summary("activation_plot_"+self.name, np.reshape(activation_image, (1, output_shape[0], output_shape[1], 1)))
        image_summary_str = sess.run(image_summary_op)
        
        SummaryWriter().writer.add_summary(image_summary_str, self.iteration)
        SummaryWriter().writer.flush()

        print("📈 activation image plotted.")

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


    # sanity checks
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
        assert self.noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def noise_validator(self, noise, allowed_noises):
        '''Validates the noise provided'''
        try:
            if noise in allowed_noises:
                return True
            elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
                t = float(noise.split('-')[1])
                if t >= 0.0 and t <= 1.0:
                    return True
                else:
                    return False
        except:
            return False
        pass

