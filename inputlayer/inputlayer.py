# -*- coding: utf-8 -*-

import numpy as np

class InputLayer:
    """
    Contains the worker thread that uses OpenCV to feed in images and video feeds to TF.
    """
    
    def assertions(self):
        #TODO make sure params are sane
        assert(42==42)
    
    def __init__(self, batch_size=1, output_size=(28,28)):
        self.batch_size = batch_size
        self.output_size = output_size
        self.assertions()
        self.callbacks = []
        print ("📸 Input Layer initalized")

    def registerCallback(self, region, callback):
        ##assert callback is a function
        self.callbacks.append([region, callback, []])
        print ("📸 callback registered")


    def deregisterCallback(self, callback):
        raise NotImplementedError()

