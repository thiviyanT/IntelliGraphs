import torch, time, yaml, os


def save_model(model, name, wandb):
    """ Save model to Weights&Biases """
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{name}.pt'))
    print('Model saved to Weights&Biases')
    print(os.path.join(wandb.run.dir, f'{name}.pt'))


def load_model(name):
    """ Load saved model """
    return torch.load(f'{name}', map_location=torch.device('cpu'))


def read_config(filename):
    """
    Read a YAML configuration file.

    :param filename: Path to the YAML configuration file.
    :return: The contents of the YAML file as a Python object.
    """
    with open(filename) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


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
