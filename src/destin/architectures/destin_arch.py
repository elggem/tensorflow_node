# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from destin.nodes import *
from destin.architectures import NetworkArchitecture

class DestinArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, node_type, node_params, receptive_field=[14,14], stride=[7,7]):
        """
         Todo here:

         1) create first layer nodes according to inputlayer output size, receptive field and stride
         2) append them to an array in the right order to be consumed for the second layer.. 
         3) ...do until theres only one node on top.

         - sess.run will evaluate only the needed computations, so there should be no
           redudant calcalations if we run all output_tensors, get the values, and run the training ops!
            http://stackoverflow.com/questions/34010987/does-tensorflow-rerun-for-each-eval-call
            
         - Autoencoder nodes need to publish their outputs to ROS themselves,
           after they have been evaluated.
        """

        self.nodes = []
        self.train_op = []




