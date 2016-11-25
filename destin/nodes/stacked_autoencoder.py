# -*- coding: utf-8 -*-

import abc
import random
import logging as log
import tensorflow as tf

from destin import SummaryWriter

class StackedAutoEncoderNode(object):
    __metaclass__ = abc.ABCMeta

    # Initialization
    def __init__(self, 
                 session, 
                 name="ae"):

        self.name = name+'-%08x' % random.getrandbits(32)
        self.session=session
        raise NotImplementedError()