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
  output_path = pjoin(os.getcwd(), 'output', path)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return output_path

_tb_pid_file = home_out(".tbpid")
_tb_path = os.path.join(os.path.dirname(tb.__file__), 'tensorboard.py')
_tb_port = "6006"

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

