# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append("./code")
from utils.logger import log

class InputLayer:
    """
    TODO doc
    """
    
    def assertions(self):
        #TODO make sure params are sane
        assert(42==42)
    
    def __init__(self, batch_size=1, output_size=(28,28)):
        self.batch_size = batch_size
        self.output_size = output_size
        self.assertions()
        self.callbacks = []
        log.debug("📸 Input Layer initalized")

    def register_callback(self, region, callback):
        #assert callback is a function
        self.callbacks.append([region, callback, []])
        log.debug("📸 callback registered")


    def deregister_callback(self, callback):
        raise NotImplementedError()

