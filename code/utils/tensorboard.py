# -*- coding: utf-8 -*-

import os
import signal
import subprocess
import sys

from tensorflow import tensorboard as tb
from summary_writer import SummaryWriter

_tb_pid_file = SummaryWriter().get_output_folder(".tbpid")
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
                        '-u', _tb_path, '--logdir={0}'.format(SummaryWriter().get_summary_folder()),
                        '--port=' + _tb_port], stdout=devnull, stderr=devnull)
  with open(_tb_pid_file, 'w') as f:
    f.write(str(p.pid))

  if False:
    subprocess.Popen(['open', 'http://localhost:{0}'.format(_tb_port)])

