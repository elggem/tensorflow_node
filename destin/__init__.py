# -*- coding: utf-8 -*-

from utils import SummaryWriter

from input import OpenCVInputLayer

from nodes import AutoEncoderNode
from nodes import StackedAutoEncoderNode

import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.DEBUG

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'yellow',
        'INFO': 'green',
        'WARNING': 'red',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    style='%'
)

logging.root.setLevel(LOG_LEVEL)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(LOG_LEVEL)
log.addHandler(stream)