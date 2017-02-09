# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf

from destin.architectures import NetworkArchitecture

class DestinArchitecture(NetworkArchitecture):

    def __init__(self, inputlayer, receptive_field=[14,14], stride=[14,14], node_params={type:"AutoEncoderNode"}):
        """
         Todo here:

         1) create nodes according to destin layout
            wire tensors up to each other
         2) 

         - sess.run will evaluate only the needed computations, so there should be no
           redudant calcalations if we run all output_tensors, get the values, and run the training ops!
            http://stackoverflow.com/questions/34010987/does-tensorflow-rerun-for-each-eval-call
            

         - Autoencoder nodes need to publish their outputs themselves,
           after they have been evaluated.

        """
        pass

