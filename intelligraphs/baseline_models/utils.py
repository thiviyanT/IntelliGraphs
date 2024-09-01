from torch import nn
from collections import Counter
import torch

def compute_entity_frequency(data):
    """
    Compute the frequency of entities in a training set of triples.

    :param data: A tensor containing triples in the form (subject, predicate, object).
    :return: A Counter object with the frequency of each entity (subject or object).
    """
    if isinstance(data, torch.Tensor):
        data = data.tolist()

    entity_frequency = Counter()
    for triple in data:
        for entity in triple:
            subject, predicate, object_ = entity
            entity_frequency[subject] += 1
            entity_frequency[object_] += 1
    return entity_frequency


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

