# -*- coding: utf-8 -*-

import datetime
import utils
import tensorflow as tf

# Singleton
class SummaryWriter(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SummaryWriter, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'writer'):
            print "📊 initializing summary writer."
            now = datetime.datetime.now()
            self.directory = utils.home_out('summaries')+now.strftime("/%Y-%m-%d-%s")
            self.writer = tf.train.SummaryWriter(self.directory)

