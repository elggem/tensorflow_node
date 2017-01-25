# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os.path
#import cv2
from skimage import transform

from destin.input import InputLayer
from sensor_msgs.msg import Image

class ROSInputLayer(InputLayer):
    """
    Contains ROS to feed in images and video feeds to TF.
    """

    def feed_topic(self, feed_callback, topic_name):

        # Callback to handle individual frames coming in via ROS
        def callback(ros_data):
            # Get numpy array from string
            np_arr = np.fromstring(ros_data.data, np.uint8).reshape(ros_data.width, ros_data.height, 3)

            # Grayscale conversion
            channels = np_arr.swapaxes(0,2)
            gray = (channels[0] + channels[1] + channels[2]) / 3 # could to different weights per channel here

            # Resize to normalized input layer size
            resized = transform.resize(gray, [self.output_size[0], self.output_size[1]]).reshape([self.output_size[0], self.output_size[1], 1])

            # Append to processing batch
            self.batch.append(resized)

            # batch is full, hand off to TF
            if len(self.batch) >= self.batch_size:
                feed_dict = {self.name + '/input:0': np.array(self.batch)}
                self.batch = []

                feed_callback(feed_dict)

                rospy.loginfo("📸 Evaluated batch")

        ## ROS subscribe...
        rospy.logwarn("subscribing")
        rospy.Subscriber(topic_name, Image, callback)
