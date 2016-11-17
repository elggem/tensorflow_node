import os
import signal
import subprocess
import sys
from os.path import join as pjoin
import datetime

import numpy as np

import tensorflow as tf
from tensorflow import tensorboard as tb



def home_out(path):
  return pjoin(os.getcwd(), 'output', path)

_tb_pid_file = home_out(".tbpid")
_tb_path = os.path.join(os.path.dirname(tb.__file__), 'tensorboard.py')
_tb_port = "6006"

##this on init.
now = datetime.datetime.now()

def get_summary_dir():
    return home_out('summaries')+now.strftime("/%Y-%m-%d-%s")


## Global summary writer.
### TODO Summary writer as singleton class.
writer = None

def get_summary_writer():
    global writer
    if (writer == None):
        #sess = tf.Session()
        writer = tf.train.SummaryWriter(get_summary_dir())

    return writer


def start_tensorboard():
  if not os.path.exists(_tb_path):
    raise EnvironmentError("tensorboard.py not found!")

  if os.path.exists(_tb_pid_file):
    tb_pid = int(open(_tb_pid_file, 'r').readline().strip())
    try:
      os.kill(tb_pid, signal.SIGKILL)
    except OSError:
      pass

    os.remove(_tb_pid_file)

  devnull = open(os.devnull, 'wb')
  p = subprocess.Popen(['nohup', sys.executable,
                        '-u', _tb_path, '--logdir={0}'.format(get_summary_dir()),
                        '--port=' + _tb_port], stdout=devnull, stderr=devnull)
  with open(_tb_pid_file, 'w') as f:
    f.write(str(p.pid))

  if False:
    subprocess.Popen(['open', 'http://localhost:{0}'.format(_tb_port)])


def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass
