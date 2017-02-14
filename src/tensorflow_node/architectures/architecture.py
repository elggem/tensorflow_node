# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from tensorflow_node.nodes import *


class NetworkArchitecture(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.train_op = tf.no_op()
        self.nodes = []
        pass

    def str_to_class(self, str):
        return getattr(sys.modules[__name__], str)

    def create_node(self, session, node_type, node_params):
        node_class = self.str_to_class(node_type)
        return node_class(session, **node_params)
