# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import os.path

from destin.input import InputLayer


class OpenCVInputLayer(InputLayer):
    """
    Contains OpenCV to feed in video feeds to TF.
    """

    def feed_to(self, feed_callback):
        
        # fixed parameters for now, could be user configurable
        frames = -1
        repeat = 0

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

        if (repeat != 0):
            self.feed_to(feed_callback)
