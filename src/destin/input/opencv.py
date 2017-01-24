# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import os.path

from destin.input import InputLayer


class OpenCVInputLayer(InputLayer):
    """
    Contains OpenCV to feed in images and video feeds to TF.
    """

    def feed_webcam(self, feed_callback, frames=-1):
        self.feed_video(feed_callback, filename=0, frames=frames)

    def feed_video(self, feed_callback, filename, frames=-1, repeat=0):
        if not os.path.isfile(filename) or filename == 0:
            raise IOError("OpenCVLayer - video file not found!")

        framecount = frames

        cap = cv2.VideoCapture(filename)

        while(framecount != 0):
            isvalid, frame = cap.read()

            if (not isvalid):
                break

            res = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            gray = gray * 1.0 / 255

            # use grayscale image
            self.batch.append(gray.reshape([self.output_size[0], self.output_size[1], 1]))

            # batch is full
            if len(self.batch) >= self.batch_size:
                feed_dict = {self.name + '/input:0': np.array(self.batch)}
                feed_callback(feed_dict)
                self.batch = []
                rospy.logdebug("📸 Evaluated frame %d" % framecount)

            if (framecount > 0):
                framecount -= 1

        cap.release()

        if (repeat != 0):
            self.feed_video(feed_callback, filename, frames=frames, repeat=repeat - 1)
