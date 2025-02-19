import pytest
import torch
from unittest.mock import patch
from typing import List, Dict, Tuple

from intelligraphs.data_loaders.loading_functions import (
    load_data_as_list,
    load_data_as_tensor
)

# we follow the arrange-act-assert practice for unit testing - https://automationpanda.com/2020/07/07/arrange-act-assert-a-pattern-for-writing-good-tests/
# assert-act-assert makes it easier to follow the unit tests, clearly laying out what the test conditions are and what is being tested

# fixtures are set up to simulate input data
@pytest.fixture
def mock_subgraphs() -> Tuple[List, List, List]:
    """Provides sample subgraphs for testing."""
    train = [
        [["entity1", "relation1", "entity2"]],
        [["entity3", "relation2", "entity4"], ["entity4", "relation1", "entity1"]]
    ]
    val = [
        [["entity1", "relation1", "entity2"]]
    ]
    test = [
        [["entity5", "relation1", "entity6"]]
    ]
    return train, val, test


@pytest.fixture
def mock_processed_data() -> Tuple[List, List, List]:
    """Provides sample processed (indexed) data."""
    train = [[[0, 0, 1]], [[2, 1, 3], [3, 0, 0]]]
    val = [[[0, 0, 1]]]
    test = [[[4, 0, 5]]]
    return train, val, test


@pytest.fixture
def mock_mappings() -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict]]:
    """Provides sample entity and relation mappings."""
    e2i = {"entity1": 0, "entity2": 1, "entity3": 2, "entity4": 3,
           "entity5": 4, "entity6": 5}
    i2e = {i: e for e, i in e2i.items()}
    r2i = {"relation1": 0, "relation2": 1}
    i2r = {i: r for r, i in r2i.items()}
    return (e2i, i2e), (r2i, i2r)


def test_load_data_as_list(mock_subgraphs, mock_processed_data, mock_mappings):
    """Test loading and processing data into list format."""
    # arrange
    dataset_name = "test_dataset"
    train_graphs, val_graphs, test_graphs = mock_subgraphs
    train_processed, val_processed, test_processed = mock_processed_data
    entity_mappings, relation_mappings = mock_mappings
    expected_edge_stats = (1, 2)
    expected_entity_stats = (2, 3)
    max_graph_length = 2

    # act
    with patch('intelligraphs.data_loaders.loading_functions.get_file_paths') as mock_get_paths, \
            patch('intelligraphs.data_loaders.loading_functions.parse_files_to_subgraphs') as mock_parse, \
            patch('intelligraphs.data_loaders.loading_functions.process_knowledge_graphs') as mock_process, \
            patch('intelligraphs.data_loaders.loading_functions.compute_statistics') as mock_stats:
        mock_get_paths.return_value = ("train.tsv", "val.tsv", "test.tsv")
        mock_parse.return_value = (train_graphs, val_graphs, test_graphs)
        mock_process.return_value = (
            train_processed,
            val_processed,
            test_processed,
            entity_mappings,
            relation_mappings,
            max_graph_length
        )
        mock_stats.return_value = (expected_edge_stats, expected_entity_stats)

        result = load_data_as_list(dataset_name)

    # assert
    train_list, valid_list, test_list, ent_map, rel_map, edge_stats, entity_stats = result
    assert train_list == train_processed
    assert valid_list == val_processed
    assert test_list == test_processed
    assert ent_map == entity_mappings
    assert rel_map == relation_mappings
    assert edge_stats == expected_edge_stats
    assert entity_stats == expected_entity_stats


def test_load_data_as_tensor(mock_processed_data, mock_mappings):
    """Test conversion of processed data into PyTorch tensors."""
    # arrange
    dataset_name = "test_dataset"
    train_processed, val_processed, test_processed = mock_processed_data
    entity_mappings, relation_mappings = mock_mappings
    edge_stats = (1, 2)
    entity_stats = (2, 3)

    # act
    with patch('intelligraphs.data_loaders.loading_functions.load_data_as_list') as mock_load:
        mock_load.return_value = (train_processed, val_processed, test_processed,
                                  entity_mappings, relation_mappings,
                                  edge_stats, entity_stats)

        train_tensor, val_tensor, test_tensor, ent_map, rel_map, e_stats, n_stats = (
            load_data_as_tensor(dataset_name))

    # assert
    assert isinstance(train_tensor, torch.Tensor)
    assert train_tensor.shape == (len(train_processed), edge_stats[1], 3)
    assert torch.all(train_tensor[0, -1] == torch.tensor([-1, -1, -1]))
    assert ent_map == entity_mappings
    assert rel_map == relation_mappings
    assert e_stats == edge_stats
    assert n_stats == entity_stats
