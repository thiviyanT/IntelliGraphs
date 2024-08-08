import torch
import time
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


def compute_entity_frequency(training_data):
    """
    Compute the frequency of entities in a training set of triples.

    :param training_data: A tensor containing triples in the form (subject, predicate, object).
    :return: A Counter object with the frequency of each entity (subject or object).
    """
    entity_frequency = Counter()
    for triple in training_data.tolist():
        for entity in triple:
            subject, predicate, object_ = entity
            entity_frequency[subject] += 1
            entity_frequency[object_] += 1
    return entity_frequency


timing_stack = []


def tic():
    """
    Start a timer by recording the current time.

    :return: None
    """
    timing_stack.append(time.time())


def toc():
    """
    Stop the timer and return the elapsed time since the last tic() call.

    :return: The elapsed time in seconds.
    """
    if len(timing_stack) == 0:
        return None
    else:
        return time.time() - timing_stack.pop()


def get_device(tensor=None):
    """
    Return the best available device for computation (CUDA or CPU).

    :param tensor:
        - If a tensor is provided, returns the device string corresponding to that tensor's device.
        - If a boolean is provided, returns 'cuda' if True, otherwise 'cpu'.
        - If no argument is provided, returns 'cuda' if available, otherwise 'cpu'.
    :return: The device string ('cuda' or 'cpu').
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(tensor, bool):
        return 'cuda' if tensor else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'
