# -*- coding: utf-8 -*-


"""

Abstract base node for DeSTIN

"""

import abc

class BaseNode(object):
    __metaclass__ = abc.ABCMeta

    # Initialization
    @abc.abstractmethod
    def __init__(self, session, name="basenode"):
        self.name = name
        self.session=session
        self.input_tensors=[]
        self.output_tensor=None
        return

    def __del__(self, input):
        return

    @abc.abstractmethod
    def initialize_graph(self):
        return

    # I/O
    def register_tensor(self, new_tensor):
        self.input_tensors.append(new_tensor)
        return

    # Persistence
    @abc.abstractmethod
    def load(self, filename):
        """Retrieve model from disk."""
        return
    
    @abc.abstractmethod
    def save(self, filename):
        """Save model to disk."""
        return