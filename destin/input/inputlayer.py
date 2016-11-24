# -*- coding: utf-8 -*-

import abc
import numpy as np
import random
import sys
import logging as log

import tensorflow as tf

class InputLayer(object):
    __metaclass__ = abc.ABCMeta
    """
    TODO doc
    """
    
    def assertions(self):
        #TODO make sure params are sane
        assert(42==42)
    
    def __init__(self, output_size=(28,28)):
        self.name = 'inputlayer-%08x' % random.getrandbits(32)
        self.output_size = output_size
        
        with tf.variable_scope(self.name) as scope:
            self.variable_scope = scope
            self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(output_size[0], output_size[1], 1), name='input')

        self.assertions()
        log.debug("📸 Input Layer initalized")

    def get_tensor_for_region(self, region):
        with tf.name_scope(self.name):
            # crop params are vertical top left, horizontal top left, height, width
            cropped = tf.image.crop_to_bounding_box(self.input_placeholder, region[1], region[0], region[3], region[2])
            flattened = tf.reshape(cropped, [-1])

        return flattened

    def get_feed_dict_for_image(self, image):
        return {self.name+'/input:0': image.reshape([28,28,1])}










    ## TODO: is this needed anymore?
    def dims_for_receiver(self, receiver):
        for region, callback in self.callbacks:
            if (callback.im_self == receiver):
                return region[2]*region[3] #return width*height

        return -1 #callback not found

