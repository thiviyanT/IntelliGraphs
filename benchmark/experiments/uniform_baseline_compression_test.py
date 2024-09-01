from intelligraphs.baseline_models import UniformBaseline
from intelligraphs.data_loaders import load_data_as_tensor


def load_data(dataset: str):
    """
    Load the dataset based on the provided name.

    :param dataset: The name of the dataset to be loaded.
    :return: A tuple containing the loaded dataset and a boolean indicating whether the dataset is synthetic.
    """
    print('Preparing the data... ', end="", flush=True)
    if 'syn-' in dataset:
        is_synthetic = True
        data = load_data_as_tensor(dataset)
    elif 'wd-' in dataset:
        is_synthetic = False
        data = load_data_as_tensor(dataset)
    else:
        raise ValueError("Unknown dataset type. Please provide a synthetic or Wikidata dataset.")
    print("(done)")
    return data, is_synthetic


if __name__ == "__main__":
    datasets = ['syn-paths', 'syn-tipr', 'syn-types', 'wd-movies', 'wd-articles']

    for dataset in datasets:
        data, is_synthetic = load_data(dataset)
        baseline = UniformBaseline(dataset=dataset, data=data, is_synthetic=is_synthetic)
        baseline.compute_bits()
