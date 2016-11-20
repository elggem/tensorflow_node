# -*- coding: utf-8 -*-

import os
import datetime
import tensorflow as tf
from os.path import join as pjoin
from logger import log

# Singleton
class SummaryWriter(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SummaryWriter, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'writer'):
            log.debug("📊 initializing summary writer.")
            now = datetime.datetime.now()
            self.directory = self.get_output_folder('summaries')+now.strftime("/%Y-%m-%d-%s")
            self.writer = tf.train.SummaryWriter(self.directory)

    def get_output_folder(self, path):
      output_path = pjoin(os.getcwd(), 'output', path)
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      return output_path

    def get_summary_folder(self):
        return directory