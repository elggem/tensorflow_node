# -*- coding: utf-8 -*-
import cv2
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
                res = cv2.resize(frame,(32, 32), interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                self.processFrame(gray)
            else:
                break

    def feedVideo(self, filename):
        cap = cv2.VideoCapture(filename)

        while(True):
            isvalid, frame = cap.read()
            if isvalid:
                res = cv2.resize(frame,(32, 32), interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                self.processFrame(gray)
            else:
                break

    def processFrame(self, frame):
        for region, callback in self.callbacks:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]
            callback(frame[y:y + h, x:x + w].flatten()/255.0)
        #iterate callbacks and split into corresponding regions
        #triggercallbacks
