#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

import tensorflow as tf
import numpy as np

from destin import AutoEncoderNode
from destin import SummaryWriter
from destin import OpenCVInputLayer

# this tests very basic functionality of the autoencoder
class AutoencoderTest(tf.test.TestCase):

    def testAE(self):
        with self.test_session() as sess:
            inputlayer = OpenCVInputLayer(output_size=(16, 16), batch_size=250)
            data = np.random.rand(250, 16, 16, 1)

            ae = AutoEncoderNode(session = sess)
            ae.register_tensor(inputlayer.get_tensor_for_region([0,0,16,16]))
            output_tensor = ae.get_output_tensor()

            result = output_tensor.eval(feed_dict={inputlayer.name+"/input:0": data})

            SummaryWriter().writer.add_graph(sess.graph)
            assert(result.shape[0] == 250)
            assert(result.shape[1] == 32)

if __name__ == '__main__':
    tf.test.main()
