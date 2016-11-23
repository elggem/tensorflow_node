# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os.path

from . import InputLayer

class OpenCVInputLayer(InputLayer):
    """
    Contains OpenCV to feed in images and video feeds to TF.
    """

    def feed_webcam(self, frames=-1):
        self.feed_video(filename=0, frames=frames)

    def feed_video(self, filename, frames=-1, repeat=0):
        if not os.path.isfile(filename):
            raise IOError("OpenCVLayer - video file not found!")

        framecount = frames
        
        cap = cv2.VideoCapture(filename)

        while(framecount != 0):
            isvalid, frame = cap.read()

            if (not isvalid):
                break

            res = cv2.resize(frame, self.output_size, interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) #TODO: allow colour input.
            self.process_frame(gray * 1.0/255)

            if (framecount > 0):
                framecount = framecount - 1

        cap.release()

        if (repeat!=0):
            self.feed_video(filename, frames=frames, repeat=repeat-1)

    def process_frame(self, frame):
        self.batch.append(frame)
        if (len(self.batch) >= self.batch_size):
            self.emit_callbacks()

    def emit_callbacks(self):
        frame_batch = np.array(self.batch)

        for region, callback in self.callbacks:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]

            callback(self, frame_batch[:, y:y + h, x:x + w].reshape((-1,w*h)))

        self.batch = []
