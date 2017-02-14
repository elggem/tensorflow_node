# -*- coding: utf-8 -*-

import abc
import sys
import rospy
import tensorflow as tf
import numpy as np

from tensorflow_node.nodes import *
from tensorflow_node.architectures import NetworkArchitecture

class DestinArchitecture(NetworkArchitecture):

    def __init__(self, session, inputlayer, node_type, node_params, receptive_field=[14,14], stride=[7,7]):
        # TODO Assertions: 
        #   - inputlayer size and receptive field / stride fit together...
        #   - ...?
    
        self.nodes = []
        self.train_op = []

        print "creating DeSTIN network..."

        def destin_node(level, number_of_layers, x_pos=0.0, y_pos=0.0):
            print " creating node @ level %i" % level
            node = self.create_node(session, node_type, node_params)

            if (level<number_of_layers):
                node.register_tensor(destin_node(level+1, number_of_layers, x_pos, y_pos))
                node.register_tensor(destin_node(level+1, number_of_layers, x_pos+stride[0], y_pos))
                node.register_tensor(destin_node(level+1, number_of_layers, x_pos, y_pos+stride[1]))
                node.register_tensor(destin_node(level+1, number_of_layers, x_pos+stride[0], y_pos+stride[1]))
            else:
                region = [int(np.round(x_pos)),int(np.round(y_pos)),receptive_field[0],receptive_field[1]]
                print "  registering region @ %i %i" % (x_pos, y_pos)
                node.register_tensor(inputlayer.get_tensor_for_region(region))

            node.initialize_graph()
            
            self.nodes.append(node)
            self.train_op.append(node.train_op)
            
            return node.get_output_tensor()

        # calculate number of levels needed...
        nr_of_layers = np.floor(np.log(np.power(inputlayer.output_size[0]/stride[0],2))/np.log(4))
        
        # create network
        destin_node(0, nr_of_layers)
        
        
        





