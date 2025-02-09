import urllib.request
import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from intelligraphs.utils import (
    load_strings,
    split_subgaphs,
    map_nodes_relations,
    pad_data,
    create_mapping,
)

from intelligraphs.data_loaders import DatasetDownloader


class CustomDataset(Dataset):
    """ Custom dataset class. """

    def __init__(self, file_path, node_mapping, relation_mapping, padding=True, max_graph_size=None):
        self.data = []

        self.data = load_strings(file_path, split_tab=True)
        self.data = split_subgaphs(self.data)
        self.data = map_nodes_relations(self.data, node_mapping, relation_mapping)
        if padding:
            if max_graph_size is None:
                raise ValueError('max_graph_size must be specified if padding is True.')
            self.data = pad_data(self.data, max_graph_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DataLoader:
    """ DataLoader for IntelliGraphs datasets. """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.base_dir = '.data'

        # Download data if not exists
        self.downloader = DatasetDownloader(download_dir=self.base_dir)
        self.downloader.download_and_verify_all()

    def load_torch(self, batch_size=32, padding=True, shuffle_train=False, shuffle_valid=False, shuffle_test=False,):
        """
        Load dataset as torch tensors for PyTorch.

        Args:
            batch_size (int): Batch size.
            padding (bool): Pad subgraphs with empty triples [-1, -1, -1].
            shuffle_train (bool): Shuffle training data.
            shuffle_valid (bool): Shuffle validation data.
            shuffle_test (bool): Shuffle test data.

        Returns:
            (train_loader, valid_loader, test_loader): PyTorch data loaders.
        """
        dataset_folder = self.dataset_name
        dataset_folder_path = os.path.join(self.base_dir, dataset_folder)

        train_file = os.path.join(dataset_folder_path, 'train_split.tsv')
        valid_file = os.path.join(dataset_folder_path, 'val_split.tsv')
        test_file = os.path.join(dataset_folder_path, 'test_split.tsv')

        (e2i, i2e), (r2i, i2r), max_len = create_mapping(train_file, valid_file, test_file)

        # Create custom datasets
        train_dataset = CustomDataset(train_file, e2i, r2i, padding=padding, max_graph_size=max_len)
        valid_dataset = CustomDataset(valid_file, e2i, r2i, padding=padding, max_graph_size=max_len)
        test_dataset = CustomDataset(test_file, e2i, r2i, padding=padding, max_graph_size=max_len)

        train_dataset = torch.tensor(train_dataset)
        valid_dataset = torch.tensor(valid_dataset)
        test_dataset = torch.tensor(test_dataset)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_valid)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

        return train_loader, valid_loader, test_loader
