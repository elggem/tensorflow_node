#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

import tensorflow as tf
import numpy as np

from destin import AutoEncoderNode
from destin import SummaryWriter
from destin import OpenCVInputLayer


# this tests the cropping of the input layer.
class InputLayerTest(tf.test.TestCase):

    def testInputLayer(self):
        with self.test_session():
            inputlayer = OpenCVInputLayer(output_size=(16, 16), batch_size=250)

            data_a = np.floor(np.random.rand(250, 8, 8, 1) * 100)
            data_b = np.floor(np.random.rand(250, 8, 8, 1) * 100)
            data_c = np.floor(np.random.rand(250, 8, 8, 1) * 100)
            data_d = np.floor(np.random.rand(250, 8, 8, 1) * 100)

            data_ab = np.concatenate([data_a, data_b], axis=1)
            data_cd = np.concatenate([data_c, data_d], axis=1)
            data_ac = np.concatenate([data_a, data_c], axis=2)
            data_bd = np.concatenate([data_b, data_d], axis=2)

            data = np.concatenate([data_ab, data_cd], axis=2)

            tensor_a = inputlayer.get_tensor_for_region([0, 0, 8, 8])
            tensor_b = inputlayer.get_tensor_for_region([8, 0, 8, 8])
            tensor_c = inputlayer.get_tensor_for_region([0, 8, 8, 8])
            tensor_d = inputlayer.get_tensor_for_region([8, 8, 8, 8])

            tensor_ab = inputlayer.get_tensor_for_region([0, 0, 16, 8])
            tensor_cd = inputlayer.get_tensor_for_region([0, 8, 16, 8])
            tensor_ac = inputlayer.get_tensor_for_region([0, 0, 8, 16])
            tensor_bd = inputlayer.get_tensor_for_region([8, 0, 8, 16])

            feed_dict = {inputlayer.name + "/input:0": data}

            return_a = tensor_a.eval(feed_dict=feed_dict)
            return_b = tensor_b.eval(feed_dict=feed_dict)
            return_c = tensor_c.eval(feed_dict=feed_dict)
            return_d = tensor_d.eval(feed_dict=feed_dict)

            return_ab = tensor_ab.eval(feed_dict=feed_dict)
            return_cd = tensor_cd.eval(feed_dict=feed_dict)
            return_ac = tensor_ac.eval(feed_dict=feed_dict)
            return_bd = tensor_bd.eval(feed_dict=feed_dict)

            reshaped_a = return_a.reshape(data_a.shape)
            reshaped_b = return_b.reshape(data_b.shape)
            reshaped_c = return_c.reshape(data_c.shape)
            reshaped_d = return_d.reshape(data_d.shape)

            reshaped_ab = return_ab.reshape(data_ab.shape)
            reshaped_cd = return_cd.reshape(data_cd.shape)
            reshaped_ac = return_ac.reshape(data_ac.shape)
            reshaped_bd = return_bd.reshape(data_bd.shape)

            # compare elementwise...
            assert((data_a == reshaped_a).all())
            assert((data_b == reshaped_b).all())
            assert((data_c == reshaped_c).all())
            assert((data_d == reshaped_d).all())

            assert((data_ab == reshaped_ab).all())
            assert((data_cd == reshaped_cd).all())

            assert((data_ac == reshaped_ac).all())
            assert((data_bd == reshaped_bd).all())


if __name__ == '__main__':
    tf.test.main()
