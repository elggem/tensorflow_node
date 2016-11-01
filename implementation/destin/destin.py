from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
from utils.data import fill_feed_dict_ae, read_data_sets_pretraining
from utils.data import read_data_sets, fill_feed_dict
from utils.flags import FLAGS
from utils.eval import loss_supervised, evaluation, do_eval_summary
from utils.utils import tile_raster_images

class DeSTIN(object):
  """Generic DeSTIN architecture.
  """
  
  def __init__(self, sess):
    """Autoencoder initializer

    Args:
      sess: tensorflow session object to use
    """

    self.__autoencoders = {}
    self.__sess = sess

    self._setup_variables()

  @property
  def session(self):
    return self.__sess

  ##TODO: define getitem, setitem for serialization

  def _setup_variables(self):
    ## TODO: setup autoencoder hierarchy here.

  def get_variables_to_init(self, n):
    return []

  def pretrain_net(self, input_pl, n, is_target=False):
    """Return net for step n training or target net

    Args:

    Returns:
      Tensor giving pretraining net or pretraining target
    """
    
    return input_pl

  def supervised_net(self, input_pl):
    """Get the supervised fine tuning net

    Args:
      input_pl: tf placeholder for ae input data
    Returns:
      Tensor giving full ae net
    """

    return input_pl


loss_summaries = {}

def training(loss, learning_rate, loss_key=None):
  """Sets up the training Ops.

  Args:

  Returns:
  """

  return False


def main_unsupervised():
  with tf.Graph().as_default() as g:
    sess = tf.Session()

  return False


def main_supervised(ae):
  with ae.session.graph.as_default():
    sess = ae.session


if __name__ == '__main__':
  destin = main_unsupervised()
  main_supervised(destin)
