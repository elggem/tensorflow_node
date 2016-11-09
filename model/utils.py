import os
import signal
import subprocess
import sys
from os.path import join as pjoin

import numpy as np

from tensorflow import tensorboard as tb

def home_out(path):
  return pjoin(os.getcwd(), 'output', path)

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
                        '-u', _tb_path, '--logdir={0}'.format(home_out('summaries')),
                        '--port=' + _tb_port], stdout=devnull, stderr=devnull)
  with open(_tb_pid_file, 'w') as f:
    f.write(str(p.pid))

  if False:
    subprocess.Popen(['open', 'http://localhost:{0}'.format(_tb_port)])

def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


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

def plot_max_activation(model, filename):
    W = model.weights['encoded']
    
    outputs = []
    
    #calculate for each hl
    for i in xrange(W.shape[1]):
        output = np.array(np.zeros(W.shape[0]),dtype='float32')
    
        W_ij_sum = 0
        for j in xrange(output.size):
            W_ij_sum += np.power(W[j][i],2)
    
        for j in xrange(output.size): 
            W_ij = W[j][i]
            output[j] = (W_ij)/(np.sqrt(W_ij_sum))
    
        outputs.append(output)
    
    #plot the results
    f, a = plt.subplots(10, 10, figsize=(10, 10))
    
    for i in range(10):
        for j in range(10):
            a[i][j].imshow(outputs[((i+1)*(j+1))-1].reshape([28,28]), cmap='Greys', interpolation="nearest")
    
    f.savefig(filename)
    #f.show()
    #plt.draw()
    #plt.waitforbuttonpress()

def plot_max_activation_fast(model, filename):
    W = model.weights['encoded']
    
    outputs = []
    
    #calculate for each hl
    for i in xrange(W.shape[1]):
        output = np.array(np.zeros(W.shape[0]),dtype='float32')
    
        W_ij_sum = 0
        for j in xrange(output.size):
            W_ij_sum += np.power(W[j][i],2)
    
        for j in xrange(output.size): 
            W_ij = W[j][i]
            output[j] = (W_ij)/(np.sqrt(W_ij_sum))
    
        outputs.append(output.reshape([28,28]))
    
    #plot the results
    #f, a = plt.subplots(10, 10, figsize=(10, 10))
    data = np.zeros([280,280], dtype=np.float32)
    
    rows = []

    for i in xrange(10):
        rows.append(np.concatenate(outputs[i*10:(i*10)+10], 0))

    data = np.concatenate(rows, 1)
    
    plt.imshow(data, cmap='Greys', interpolation="nearest")
    plt.savefig(filename)
