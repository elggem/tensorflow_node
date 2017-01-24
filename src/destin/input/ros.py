# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os.path
import cv2


from destin.input import InputLayer
from sensor_msgs.msg import Image

class ROSInputLayer(InputLayer):
    """
    Contains ROS to feed in images and video feeds to TF.
    """




    def feed_topic(self, feed_callback, topic_name):
        rospy.logwarn("hio")

        def callback(ros_data):
            # use grayscale image
            np_arr = np.fromstring(ros_data.data, np.uint8).reshape(ros_data.width, ros_data.height, 3)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print np_arr

            self.batch.append(image.reshape([self.output_size[0], self.output_size[1], 1]))
            
            # batch is full
            if len(self.batch) >= self.batch_size:
                feed_dict = {self.name + '/input:0': np.array(self.batch)}
                feed_callback(feed_dict)
                self.batch = []
                rospy.logdebug("📸 Evaluated frame %d" % framecount)

        ## ROS subscribe...
        rospy.logwarn("subscribing")
        rospy.Subscriber(topic_name, Image, callback)
