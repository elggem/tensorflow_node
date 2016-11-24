# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os.path

from destin.input import InputLayer

class OpenCVInputLayer(InputLayer):
    """
    Contains OpenCV to feed in images and video feeds to TF.
    """

    def feed_webcam(self, tensor, frames=-1):
        self.feed_video(tensor, filename=0, frames=frames)

    def feed_video(self, tensor, filename, frames=-1, repeat=0):
        if not os.path.isfile(filename) or filename == 0:
            raise IOError("OpenCVLayer - video file not found!")

        framecount = frames
        
        cap = cv2.VideoCapture(filename)

        while(framecount != 0):
            isvalid, frame = cap.read()

            if (not isvalid):
                break

            res = cv2.resize(frame, self.output_size, interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) #TODO: allow colour input.
            gray = gray * 1.0/255

            #use gray
            tensor.eval(feed_dict=self.get_feed_dict_for_image(gray))

            if (framecount > 0):
                framecount -= 1

        cap.release()

        if (repeat!=0):
            self.feed_video(filename, frames=frames, repeat=repeat-1)

