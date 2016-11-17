# -*- coding: utf-8 -*-
import cv2
import numpy as np
from . import InputLayer

class OpenCVInputLayer(InputLayer):
    """
    Contains the worker thread that uses OpenCV to feed in images and video feeds to TF.
    """

    def feedWebcam(self):
        cap = cv2.VideoCapture(0)

        while(True):
            isvalid, frame = cap.read()
            if isvalid:
                res = cv2.resize(frame, self.output_size, interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                self.processFrame(gray)
            else:
                break

    def feedVideo(self, filename, frames=-1):
        cap = cv2.VideoCapture(filename)

        while(frames != 0):
            isvalid, frame = cap.read()
            if isvalid:
                res = cv2.resize(frame, self.output_size, interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                self.processFrame(gray)
            else:
                break

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
