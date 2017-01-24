#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as log

import rospy
import tensorflow as tf
import numpy as np

from destin import AutoEncoderNode
from destin import SummaryWriter

log.info("Hello!")
log.info("recording summaries to " + SummaryWriter().get_summary_folder())
