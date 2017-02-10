# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf


class NetworkArchitecture(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.train_op = tf.no_op()
        self.nodes = []
        pass
