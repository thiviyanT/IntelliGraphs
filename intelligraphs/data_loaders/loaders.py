from typing import Tuple, List, Dict, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from intelligraphs.data_loaders.utils import (
    read_tsv_file,
    parse_subgraphs,
    convert_to_indices,
    pad_to_max_length,
    process_knowledge_graphs,
)
from intelligraphs.data_loaders import DatasetDownloader

DataMapping = Tuple[Dict[str, int], Dict[int, str]]


class CustomDataset(Dataset):
    """Custom dataset class."""

    def __init__(
            self,
            file_path: str,
            node_mapping: Dict[str, int],
            relation_mapping: Dict[str, int],
            padding: bool = True,
            max_graph_size: Optional[int] = None
    ) -> None:
        if padding and max_graph_size is None:
            raise ValueError('max_graph_size must be specified if padding is True.')

        self.data = self._process_data(
            file_path,
            node_mapping,
            relation_mapping,
            padding,
            max_graph_size
        )

    def _process_data(
            self,
            file_path: str,
            node_mapping: Dict[str, int],
            relation_mapping: Dict[str, int],
            padding: bool,
            max_graph_size: Optional[int]
    ) -> List:
        data = read_tsv_file(file_path, split_tab=True)
        data = parse_subgraphs(data)
        data = convert_to_indices(data, node_mapping, relation_mapping)

        if padding and max_graph_size:
            data = pad_to_max_length(data, max_graph_size)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List:
        return self.data[index]


class DataLoader:
    """DataLoader for IntelliGraphs datasets."""

    def __init__(
            self,
            dataset_name: str,
            base_dir: str = '.data',
            automatic_download: bool = False
    ) -> None:
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.entity_to_id = {}
        self.relation_to_id = {}

        if automatic_download:
            self._download_dataset()

    def __repr__(self) -> str:
        """Return a string representation of the DataLoader."""
        is_loaded = len(self.entity_to_id) > 0 and len(self.relation_to_id) > 0
        status = "loaded" if is_loaded else "not loaded"
        if is_loaded:
            return (f"IntelliGraphs DataLoader(dataset='{self.dataset_name}', "
                    f"status='{status}', "
                    f"base_dir='{self.base_dir}', "
                    f"entities={len(self.entity_to_id)}, "
                    f"relations={len(self.relation_to_id)})")
        else:
            return (f"IntelliGraphs DataLoader(dataset='{self.dataset_name}', "
                    f"status='{status}', "
                    f"base_dir='{self.base_dir}')")

    def _download_dataset(self) -> None:
        """Download dataset if it doesn't exist."""
        downloader = DatasetDownloader(download_dir=self.base_dir)
        downloader.download_and_verify_all()

    def _get_file_paths(self) -> Tuple[str, str, str]:
        """Get paths for train, validation and test files."""
        dataset_folder_path = os.path.join(self.base_dir, self.dataset_name)
        return (
            os.path.join(dataset_folder_path, 'train_split.tsv'),
            os.path.join(dataset_folder_path, 'val_split.tsv'),
            os.path.join(dataset_folder_path, 'test_split.tsv')
        )

    def _create_datasets(
            self,
            file_paths: Tuple[str, str, str],
            node_mapping: Dict[str, int],
            relation_mapping: Dict[str, int],
            padding: bool,
            max_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train, validation and test datasets."""
        train_file, valid_file, test_file = file_paths

        datasets = [
            CustomDataset(
                file_path,
                node_mapping,
                relation_mapping,
                padding=padding,
                max_graph_size=max_len
            )
            for file_path in [train_file, valid_file, test_file]
        ]

        return tuple(torch.tensor(dataset) for dataset in datasets)

    def _create_dataloaders(
            self,
            datasets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_size: int,
            shuffle_params: Tuple[bool, bool, bool]
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """Create train, validation and test dataloaders."""
        return tuple(
            TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )
            for dataset, shuffle in zip(datasets, shuffle_params)
        )

    def load_torch(
            self,
            batch_size: int = 32,
            padding: bool = True,
            shuffle_train: bool = False,
            shuffle_valid: bool = False,
            shuffle_test: bool = False
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        Load dataset as torch tensors for PyTorch.

        Args:
            batch_size: Batch size for dataloaders
            padding: Whether to pad subgraphs with empty triples [-1, -1, -1]
            shuffle_train: Shuffle training data
            shuffle_valid: Shuffle validation data
            shuffle_test: Shuffle test data

        Returns:
            Tuple of (train_loader, valid_loader, test_loader)
        """
        file_paths = self._get_file_paths()
        _, _, _, entity_mappings, relation_mappings, max_length = process_knowledge_graphs(*file_paths)

        self.entity_to_id = entity_mappings[0]  # entity_to_index mapping
        self.relation_to_id = relation_mappings[0]  # relation_to_index mapping

        datasets = self._create_datasets(
            file_paths,
            self.entity_to_id,
            self.relation_to_id,
            padding,
            max_length
        )

        return self._create_dataloaders(
            datasets,
            batch_size,
            (shuffle_train, shuffle_valid, shuffle_test)
        )
