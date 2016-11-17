#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import cv
import numpy as np
import model.utils as utils
from os.path import join as pjoin

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

train_data = mnist.train.images 

print "👉 processed input data!"

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('./mnist.mjpeg',cv.FOURCC(*'MJPG'), 200, (28,28))

i = 0

for frame in train_data:
    print "frame... " + str(i)
    i = i + 1
    frame = frame * 255.0
    x = frame.reshape([28,28]).astype('uint8')
    x = np.repeat(x,3,axis=1)
    x = x.reshape(28, 28, 3)
    out.write(x)

# Release everything if job is finished
out.release()