import pytest
from unittest.mock import Mock, patch, mock_open
import torch
from typing import Dict, List, Tuple

from intelligraphs.data_loaders.loaders import CustomDataset, DataLoader


@pytest.fixture
def mock_file_content() -> str:
    return (
        "entity1\trelation1\tentity2\n"
        "entity3\trelation2\tentity4\n"
        "\n"
        "entity5\trelation1\tentity6\n"
    )


@pytest.fixture
def mock_mappings() -> Tuple[Dict[str, int], Dict[str, int]]:
    node_mapping = {
        "entity1": 0, "entity2": 1,
        "entity3": 2, "entity4": 3,
        "entity5": 4, "entity6": 5
    }
    relation_mapping = {"relation1": 0, "relation2": 1}
    return node_mapping, relation_mapping


@pytest.fixture
def processed_data() -> List[List[List[int]]]:
    return [
        [[0, 0, 1], [2, 1, 3]],
        [[4, 0, 5]]
    ]


# Tests for CustomDataset

def test_custom_dataset_init(mock_mappings):
    # arrange
    node_mapping, relation_mapping = mock_mappings

    # act & assert
    with pytest.raises(ValueError):
        # Should raise error when padding=True but no max_graph_size
        CustomDataset("dummy.tsv", node_mapping, relation_mapping, padding=True)


def test_custom_dataset_len(mock_file_content, mock_mappings):
    # arrange
    node_mapping, relation_mapping = mock_mappings

    # act
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('intelligraphs.data_loaders.utils.read_tsv_file') as mock_read:
            mock_read.return_value = mock_file_content.split('\n')
            dataset = CustomDataset(
                "dummy.tsv",
                node_mapping,
                relation_mapping,
                padding=True,
                max_graph_size=3
            )

    # assert
    assert len(dataset) > 0


def test_custom_dataset_getitem(mock_file_content, mock_mappings, processed_data):
    # arrange
    node_mapping, relation_mapping = mock_mappings

    # act
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('intelligraphs.data_loaders.utils.read_tsv_file') as mock_read:
            mock_read.return_value = mock_file_content.split('\n')
            dataset = CustomDataset(
                "dummy.tsv",
                node_mapping,
                relation_mapping,
                padding=True,
                max_graph_size=3
            )
            dataset.data = processed_data

    # assert
    item = dataset[0]
    assert isinstance(item, list)
    assert len(item) > 0
    assert all(isinstance(triple, list) for triple in item)
    assert all(len(triple) == 3 for triple in item)


# Tests for DataLoader
def test_dataloader_init():
    # arrange & act
    loader = DataLoader("test_dataset")

    # assert
    assert loader.dataset_name == "test_dataset"
    assert loader.base_dir == ".data"


def test_get_file_paths():
    # arrange
    loader = DataLoader("test_dataset")

    # act
    train_path, valid_path, test_path = loader._get_file_paths()

    # assert
    assert train_path.endswith('train_split.tsv')
    assert valid_path.endswith('val_split.tsv')
    assert test_path.endswith('test_split.tsv')
    assert all(path.startswith('.data/test_dataset') for path in [train_path, valid_path, test_path])


def test_create_dataloaders():
    """Test dataloader creation functionality."""
    # arrange
    loader = DataLoader("test_dataset")
    mock_datasets = (torch.tensor([[[1, 1, 1]]]),
                     torch.tensor([[[2, 2, 2]]]),
                     torch.tensor([[[3, 3, 3]]]))
    batch_size = 16
    shuffle_params = (True, False, True)

    # mock
    with patch('intelligraphs.data_loaders.loaders.TorchDataLoader') as mock_dataloader_class:
        mock_dataloaders = [Mock() for _ in range(3)]
        mock_dataloader_class.side_effect = mock_dataloaders

        # act
        dataloaders = loader._create_dataloaders(
            mock_datasets,
            batch_size,
            shuffle_params
        )

    # assert
    assert len(dataloaders) == 3
    assert mock_dataloader_class.call_count == 3

    # verify correct parameters were passed to each DataLoader
    for i, (dataset, shuffle) in enumerate(zip(mock_datasets, shuffle_params)):
        args, kwargs = mock_dataloader_class.call_args_list[i]
        assert args[0] is dataset
        assert kwargs['batch_size'] == batch_size
        assert kwargs['shuffle'] == shuffle


def test_load_torch():
    """Test full data loading pipeline."""
    # arrange
    dataset_name = "test_dataset"
    batch_size = 64
    shuffle_train, shuffle_valid, shuffle_test = True, False, False

    loader = DataLoader(dataset_name)

    # mock
    with patch.object(loader, '_get_file_paths') as mock_get_paths, \
            patch('intelligraphs.data_loaders.loaders.process_knowledge_graphs') as mock_process, \
            patch.object(loader, '_create_datasets') as mock_create_datasets, \
            patch.object(loader, '_create_dataloaders') as mock_create_dataloaders:
        # Set up return values
        file_paths = ('.data/test/train.tsv', '.data/test/val.tsv', '.data/test/test.tsv')
        mock_get_paths.return_value = file_paths

        e2i = {"entity1": 0, "entity2": 1}
        i2e = {0: "entity1", 1: "entity2"}
        r2i = {"relation1": 0}
        i2r = {0: "relation1"}
        max_len = 10
        mock_process.return_value = ((e2i, i2e), (r2i, i2r), max_len)

        mock_datasets = (torch.tensor([[[1, 1, 1]]]),
                         torch.tensor([[[2, 2, 2]]]),
                         torch.tensor([[[3, 3, 3]]]))
        mock_create_datasets.return_value = mock_datasets

        expected_dataloaders = (Mock(), Mock(), Mock())
        mock_create_dataloaders.return_value = expected_dataloaders

        # Act
        dataloaders = loader.load_torch(
            batch_size=batch_size,
            padding=True,
            shuffle_train=shuffle_train,
            shuffle_valid=shuffle_valid,
            shuffle_test=shuffle_test
        )

    # assert
    mock_get_paths.assert_called_once()
    mock_process.assert_called_once_with(*file_paths)

    # verify datasets were created with the right parameters
    mock_create_datasets.assert_called_once_with(
        file_paths,
        e2i,
        r2i,
        True,  # padding
        max_len
    )

    # verify dataloaders were created with the right parameters
    mock_create_dataloaders.assert_called_once_with(
        mock_datasets,
        batch_size,
        (shuffle_train, shuffle_valid, shuffle_test)
    )

    # verify the returned dataloaders are what we expected
    assert dataloaders == expected_dataloaders
