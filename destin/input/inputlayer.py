# -*- coding: utf-8 -*-

import numpy as np
import random
import sys
import logging as log

class InputLayer:
    """
    TODO doc
    """
    
    def assertions(self):
        #TODO make sure params are sane
        assert(42==42)
    
    def __init__(self, batch_size=1, output_size=(28,28)):
        self.name = 'inputlayer-%08x' % random.getrandbits(32)
        self.batch = []
        self.batch_size = batch_size
        self.output_size = output_size
        self.assertions()
        self.callbacks = []
        log.debug("📸 Input Layer initalized")

    def register_callback(self, region, callback):
        #assert callback is a function
        self.callbacks.append([region, callback])
        log.debug("📸 callback registered")

    def dims_for_receiver(self, receiver):
        for region, callback in self.callbacks:
            if (callback.im_self == receiver):
                return region[2]*region[3] #return width*height

        return -1 #callback not found

    def deregister_callback(self, callback):
        raise NotImplementedError()

