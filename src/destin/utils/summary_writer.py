# -*- coding: utf-8 -*-

import rospy
import os
import datetime
import tensorflow as tf
import numpy as np
from os.path import join as pjoin

import rospy

# Singleton
class SummaryWriter(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SummaryWriter, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'writer'):
            rospy.logdebug("initializing summary writer.")
            now = datetime.datetime.now()
            self.directory = self.get_output_folder('summaries') + now.strftime("/%Y-%m-%d-%s")
            self.writer = tf.train.SummaryWriter(self.directory)

    def get_output_folder(self, path):
        #output_path = pjoin(os.getcwd(), 'output', path)
        output_path = rospy.get_param("publishing/summary_folder")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def get_summary_folder(self):
        return self.directory

    def batch_of_1d_to_image_grid(self, batch):
        # TODO refactor
        batch = np.array(batch)

        data_wh = int(np.ceil(np.power(batch.shape[1], 0.5)))
        data_shape = [data_wh, data_wh]

        output_grid_wh = int(np.ceil(np.power(batch.shape[0], 0.5)))

        output_rows = []
        outputs = []

        for data in batch:
            data -= data.min()
            data *= 1.0 / data.max()

            z_pad = np.zeros((data_wh * data_wh) - len(data))

            if (len(z_pad) > 0):
                data = np.concatenate(data, z_pad)

            image = data.reshape(data_shape)
            image = np.pad(image, pad_width=(1, 1), mode='constant', constant_values=0)
            outputs.append(image)

        while len(outputs) < (output_grid_wh * output_grid_wh):
            outputs.append(np.zeros([data_shape[0] + 2, data_shape[1] + 2]))

        for i in xrange(output_grid_wh):
            output_rows.append(np.concatenate(outputs[i * output_grid_wh:(i * output_grid_wh) + output_grid_wh], 0))

        activation_image = np.concatenate(output_rows, 1)

        return activation_image

    def image_summary(self, tag, image):
        image = image.reshape((1, image.shape[0], image.shape[1], 1)).astype(np.float32)

        image_summary_op = tf.image_summary(tag, image)
        image_summary_str = tf.Session().run(image_summary_op)

        SummaryWriter().writer.add_summary(image_summary_str, 0)
        SummaryWriter().writer.flush()

        rospy.loginfo("📈 " + tag + " image plotted.")
        pass
