import pytest
from unittest.mock import mock_open, patch
import warnings
from typing import List, Dict, Tuple, Union

from intelligraphs.data_loaders.utils import (
    get_file_paths,
    read_tsv_file,
    convert_to_indices,
    pad_to_max_length,
    parse_subgraphs,
    parse_files_to_subgraphs,
    create_mappings_from_subgraphs,
    process_knowledge_graphs,
    compute_statistics,
    compute_min_max_edges,
    compute_min_max_entities
)

# we follow the arrange-act-assert practice for unit testing - https://automationpanda.com/2020/07/07/arrange-act-assert-a-pattern-for-writing-good-tests/
# assert-act-assert makes it easier to follow the unit tests, clearly laying out what the test conditions are and what is being tested

# fixtures are set up to simulate input data
@pytest.fixture
def sample_tsv_content() -> str:
    """Provides a sample TSV file content with subgraphs separated by empty lines."""
    return (
        "entity1\trelation1\tentity2\n"
        "entity3\trelation2\tentity4\n"
        "\n"
        "entity5\trelation1\tentity6"
    )

@pytest.fixture
def sample_subgraphs() -> List[List[List[str]]]:
    """Provides sample subgraphs structure for testing."""
    return [
        [["entity1", "relation1", "entity2"]],
        [["entity3", "relation2", "entity4"],
         ["entity4", "relation1", "entity1"]]
    ]

@pytest.fixture
def sample_mappings() -> Tuple[Dict[str, int], Dict[str, int]]:
    """Provides sample entity and relation mappings."""
    e2i = {"entity1": 0, "entity2": 1, "entity3": 2, "entity4": 3}
    r2i = {"relation1": 0, "relation2": 1}
    return e2i, r2i


def test_get_file_paths():
    """Test construction of file paths for dataset splits."""
    # arrange
    dataset_name = "test_dataset"
    data_dir = "test_data"
    expected_paths = (
        "test_data/test_dataset/train_split.tsv",
        "test_data/test_dataset/val_split.tsv",
        "test_data/test_dataset/test_split.tsv"
    )

    # act
    result = get_file_paths(dataset_name, data_dir)

    # assert
    assert result == expected_paths
    assert all(path.endswith('.tsv') for path in result)


def test_read_tsv_file_empty_file():
    """Read empty TSV file."""
    # arrange
    empty_file = ""

    # act
    with patch('builtins.open', mock_open(read_data=empty_file)):
        result = read_tsv_file('dummy.tsv', split_tab=True)

    # assert
    assert result == []


def test_read_tsv_file(sample_tsv_content):
    """Test reading TSV file with tab-separated values."""
    # arrange
    expected_lines = [
        ['entity1', 'relation1', 'entity2'],
        ['entity3', 'relation2', 'entity4'],
        [''],
        ['entity5', 'relation1', 'entity6']
    ]

    # act
    with patch('builtins.open', mock_open(read_data=sample_tsv_content)):
        result = read_tsv_file('dummy.tsv', split_tab=True)

    # assert
    assert result == expected_lines


def test_parse_subgraphs_with_empty_separator():
    """Test parsing raw file content into subgraphs."""
    # arrange
    input_lines = [
        ['entity1', 'relation1', 'entity2'],
        [''],
        ['entity3', 'relation2', 'entity4']
    ]
    expected_subgraphs = [
        [['entity1', 'relation1', 'entity2']],
        [['entity3', 'relation2', 'entity4']]
    ]

    # act
    result = parse_subgraphs(input_lines)

    # assert
    assert result == expected_subgraphs


def test_create_mappings_from_subgraphs(sample_subgraphs):
    """Test creation of entity and relation mappings from subgraphs."""
    # arrange
    expected_entity_count = 4  # unique entities in sample_subgraphs
    expected_relation_count = 2  # unique relations in sample_subgraphs

    # act
    (e2i, i2e), (r2i, i2r) = create_mappings_from_subgraphs(sample_subgraphs)

    # assert
    assert len(e2i) == expected_entity_count
    assert len(r2i) == expected_relation_count
    assert all(i2e[e2i[e]] == e for e in e2i)
    assert all(i2r[r2i[r]] == r for r in r2i)


def test_convert_to_indices(sample_subgraphs, sample_mappings):
    """Test conversion of string-based subgraphs to integer indices."""
    # arrange
    e2i, r2i = sample_mappings
    expected_first_triple = [0, 0, 1]  # [entity1, relation1, entity2]

    # act
    result = convert_to_indices(sample_subgraphs, e2i, r2i)

    # assert
    assert len(result) == len(sample_subgraphs)
    assert result[0][0] == expected_first_triple


def test_parse_files_to_subgraphs(sample_tsv_content):
    """Test complete pipeline of reading and parsing files into subgraphs."""
    # arrange
    expected_subgraph_counts = (2, 2, 2)  # Two subgraphs per file

    # act
    with patch('builtins.open', mock_open(read_data=sample_tsv_content)):
        train_graphs, val_graphs, test_graphs = parse_files_to_subgraphs(
            'train.tsv', 'val.tsv', 'test.tsv'
        )

    # assert
    assert len(train_graphs) == expected_subgraph_counts[0]
    assert len(val_graphs) == expected_subgraph_counts[1]
    assert len(test_graphs) == expected_subgraph_counts[2]


def test_pad_to_max_length():
    """Test padding subgraphs to uniform length."""
    # arrange
    subgraphs = [[[1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]
    target_length = 3
    padding_triple = [-1, -1, -1]

    # act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pad_to_max_length(subgraphs, target_length)

    # assert
    assert all(len(subgraph) == target_length for subgraph in result)
    assert result[0][-1] == padding_triple


def test_process_knowledge_graphs(sample_tsv_content):
    """Test the complete knowledge graph processing pipeline."""
    # arrange
    expected_entity_count = 6  # unique entities in sample data
    expected_relation_count = 2  # unique relations in sample data

    # act
    with patch('builtins.open', mock_open(read_data=sample_tsv_content)):
        (train_idx, val_idx, test_idx,
         entity_mappings, relation_mappings,
         max_length) = process_knowledge_graphs(
            'train.tsv', 'val.tsv', 'test.tsv'
        )

    # assert
    e2i, i2e = entity_mappings
    r2i, i2r = relation_mappings
    assert len(e2i) == expected_entity_count
    assert len(r2i) == expected_relation_count
    assert max_length > 0


def test_compute_min_max_edges(sample_subgraphs):
    """Test computation of minimum and maximum edges in subgraphs."""
    # arrange
    expected_min = 1  # First subgraph has 1 edge
    expected_max = 2  # Second subgraph has 2 edges

    # act
    min_edges, max_edges = compute_min_max_edges(sample_subgraphs)

    # assert
    assert min_edges == expected_min
    assert max_edges == expected_max


def test_compute_min_max_entities(sample_subgraphs):
    """Test computation of minimum and maximum entities in subgraphs."""
    # arrange
    expected_min = 2  # First subgraph has 2 entities
    expected_max = 3  # Second subgraph has 3 entities

    # act
    min_entities, max_entities = compute_min_max_entities(sample_subgraphs)

    # assert
    assert min_entities == expected_min
    assert max_entities == expected_max


def test_compute_statistics(sample_subgraphs):
    """Test computation of combined edge and entity statistics."""
    # arrange
    train = sample_subgraphs
    val = sample_subgraphs[:1]  # Use first subgraph for validation
    test = sample_subgraphs[:1]  # Use first subgraph for test

    # act
    (min_edges, max_edges), (min_entities, max_entities) = compute_statistics(
        train, val, test
    )

    # assert
    assert min_edges == 1  # Smallest subgraph has 1 edge
    assert max_edges == 2  # Largest subgraph has 2 edges
    assert min_entities == 2  # Smallest subgraph has 2 entities
    assert max_entities == 3  # Largest subgraph has 3 entities

