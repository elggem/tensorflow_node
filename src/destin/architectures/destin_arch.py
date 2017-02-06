# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf


class DestinArchitecture(NetworkArchitecture):

    def __init__(self, inputlayer, receptive_field=[14,14], stride=[14,14], node_params={type:"AutoEncoderNode"}):
        pass

