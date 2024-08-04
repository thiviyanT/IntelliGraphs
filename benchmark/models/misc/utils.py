import torch
from torch import nn
from collections import Counter
import yaml


def read_config(filename):
    """
    Read a YAML configuration file.

    :param filename: Path to the YAML configuration file.
    :return: The contents of the YAML file as a Python object.
    """
    with open(filename) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def compute_entity_frequency(train):
    """
    Compute the frequency of entities in a training set of triples.

    :param train: A list or array of triples, where each triple is a tuple of (subject, predicate, object).
    :return: A Counter object containing the frequency of each entity in the training set.
    """
    frq = Counter()
    for t in train.tolist():
        for e in t:
            s, p, o = e
            frq[s] += 1
            frq[o] += 1
    return frq


def initialize(tensor, method):
    """
    Initialize a tensor using a specified initialization method.

    :param tensor: The tensor to be initialized.
    :param method: The initialization method to use. Supported methods are:
                   'uniform', 'glorot_normal', 'glorot_uniform', 'normal'.
    :raise Exception: If an unrecognized initialization method is provided.
    """
    if method == 'uniform':
        nn.init.uniform_(tensor, -1, 1)
    elif method == 'glorot_normal':
        nn.init.xavier_normal_(tensor, gain=1)
    elif method == 'glorot_uniform':
        nn.init.xavier_uniform_(tensor, gain=1)
    elif method == 'normal':
        nn.init.normal_(tensor, 0, 1)
    else:
        raise Exception(f'Initialization method {method} not recognized.')


def get_device(tensor=None):
    """
    Get the device string for the best available device or the device corresponding to the input tensor.

    :param tensor: A tensor or boolean value. If None, the best available device ('cuda' or 'cpu') is returned.
                  If a boolean, returns 'cuda' if True, otherwise 'cpu'.
                  If a tensor, returns 'cuda' if the tensor is on a CUDA device, otherwise 'cpu'.
    :return: A string representing the device ('cuda' or 'cpu').
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if type(tensor) == bool:
        return 'cuda'if tensor else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'
