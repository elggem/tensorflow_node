# -*- coding: utf-8 -*-

import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.INFO

#LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
#formatter = ColoredFormatter(LOGFORMAT)

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

logging.root.setLevel(LOG_LEVEL)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

log = logging.getLogger()

if not len(log.handlers):
    log.setLevel(LOG_LEVEL)
    log.addHandler(stream)


## TODO: file output

#log.debug("A quirky message only developers care about")
#log.info("Curious users might want to know this")
#log.warn("Something is wrong and any user should be informed")
#log.error("Serious stuff, this is red for a reason")
#log.critical("OH NO everything is on fire")

#import this as:
# from utils.logger import log