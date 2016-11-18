# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os.path

from . import InputLayer

class OpenCVInputLayer(InputLayer):
    """
    Contains OpenCV to feed in images and video feeds to TF.
    """

    def feedWebcam(self, frames=-1):
        self.feedVideo(filename=0, frames=frames)

    def feedVideo(self, filename, frames=-1):
        if not os.path.isfile(filename):
            raise IOError("OpenCVLayer - video file not found!")

        cap = cv2.VideoCapture(filename)

        while(frames != 0):
            isvalid, frame = cap.read()

            if (not isvalid):
                break

            res = cv2.resize(frame, self.output_size, interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) #TODO: allow colour input.
            self.processFrame(gray)

            if (frames > 0):
                frames = frames - 1

    def processFrame(self, frame):
        for region, callback, batch in self.callbacks:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]
            roi = frame[y:y + h, x:x + w].flatten()/255.0

            batch.append(roi)
            
            if (len(batch) >= self.batch_size):
                callback(np.array(batch))
                batch[:] = []
