# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import os.path
import rospy

from tensorflow_node.input import InputLayer


class OpenCVInputLayer(InputLayer):
    """
    Contains OpenCV to feed in video feeds to TF.
    """

    def __init__(self, batch_size=1, output_size=[28, 28], input="", number_of_frames=-1, repeat=True):
        super(OpenCVInputLayer, self).__init__(batch_size, output_size, input)
        self.number_of_frames = number_of_frames
        self.repeat = repeat

    def feed_to(self, feed_callback):
        
        # TODO: there should be clearer distinction here, get these params via daemon.
        frames = rospy.get_param("tensorflow_node/inputlayer/params/number_of_frames")
        repeat = rospy.get_param("tensorflow_node/inputlayer/params/repeat")

        # check if file exists
        if not os.path.isfile(self.input) or self.input == 0:
            raise IOError("OpenCVLayer - video file not found!")

        cap = cv2.VideoCapture(self.input)

        while(frames != 0):
            isvalid, frame = cap.read()

            if (not isvalid):
                break

            res = cv2.resize(frame, (self.output_size[0], self.output_size[1]), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            gray = gray * 1.0 / 255

            # use grayscale image
            self.batch.append(gray.reshape([self.output_size[0], self.output_size[1], 1]))

            # batch is full

            # Can we use TF Queue for this?

            if len(self.batch) >= self.batch_size:
                feed_dict = {self.name + '/input:0': np.array(self.batch)}
                feed_callback(feed_dict)
                self.batch = []
                print("Inputlayer: Evaluated batch of size %i" % self.batch_size)

            if (frames > 0):
                frames -= 1

        cap.release()

        if (repeat):
            self.feed_to(feed_callback)
