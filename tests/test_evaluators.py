import pytest
from typing import List, Tuple, Dict
from intelligraphs.evaluators import (
    check_semantics, is_graph_empty, is_graph_in_training_data, validate_graph, compile_results
)

# Sample data and mock functions for testing
sampled_graphs = [
    [("A", "r1", "B"), ("B", "r2", "C")],
    [("X", "r3", "Y"), ("Y", "r4", "Z")],
    [],
    [("A", "r1", "C"), ("C", "r2", "D")]
]

training_data = [
    [("A", "r1", "B"), ("B", "r2", "C")],
    [("M", "r5", "N"), ("N", "r6", "O")]
]


def mock_semantic_check_func(graph, entity_labels=None, relation_labels=None, length=None):
    """Mock function to simulate semantic validity check."""
    if length is not None and len(graph) != length:
        return False
    # Consider any graph with subject "A" and object "C" as invalid
    for triple in graph:
        if triple[0] == "A" and triple[2] == "C":
            return False
    return True


entity_labels = {0: "A", 1: "B", 2: "C"}
relation_labels = {0: "r1", 1: "r2", 2: "r3"}

expected_graph_size = 2


# Test functions

def test_is_graph_empty():
    assert is_graph_empty([]) == True
    assert is_graph_empty([("A", "r1", "B")]) == False


def test_is_graph_in_training_data():
    assert is_graph_in_training_data([("A", "r1", "B"), ("B", "r2", "C")], training_data) == True
    assert is_graph_in_training_data([("X", "r3", "Y"), ("Y", "r4", "Z")], training_data) == False


def test_validate_graph():
    valid, valid_but_wrong_size = validate_graph(
        [("A", "r1", "B"), ("B", "r2", "C")],
        mock_semantic_check_func,
        entity_labels,
        relation_labels,
        expected_graph_size
    )
    assert valid == True
    assert valid_but_wrong_size == False

    valid, valid_but_wrong_size = validate_graph(
        [("A", "r1", "B")],
        mock_semantic_check_func,
        entity_labels,
        relation_labels,
        expected_graph_size
    )
    assert valid == False
    assert valid_but_wrong_size == True

    valid, valid_but_wrong_size = validate_graph(
        [("A", "r1", "C")],
        mock_semantic_check_func,
        entity_labels,
        relation_labels,
        expected_graph_size
    )
    assert valid == False
    assert valid_but_wrong_size == False


def test_compile_results():
    results, categorized_graphs = compile_results(
        total_graphs=4,
        valid_graph_count=2,
        valid_novel_graph_count=1,
        valid_but_wrong_size_count=1,
        novel_graph_count=2,
        original_graph_count=1,
        empty_graph_count=1,
        valid_graphs=[sampled_graphs[0], sampled_graphs[3]],
        valid_novel_graphs=[sampled_graphs[1]],
        invalid_graphs=[sampled_graphs[2]]
    )

    assert results['pct_semantically_valid'] == 50.0
    assert results['pct_valid_novel_graphs'] == 25.0
    assert results['pct_novel_graphs'] == 50.0
    assert results['pct_original_graphs'] == 25.0
    assert results['pct_empty_graphs'] == 25.0
    assert results['pct_valid_but_wrong_size'] == 25.0

    assert categorized_graphs['valid_graphs'] == [sampled_graphs[0], sampled_graphs[3]]
    assert categorized_graphs['valid_novel_graphs'] == [sampled_graphs[1]]
    assert categorized_graphs['invalid_graphs'] == [sampled_graphs[2]]


def test_check_semantics():
    results, categorized_graphs = check_semantics(
        sampled_graphs=sampled_graphs,
        training_data=training_data,
        semantic_check_func=mock_semantic_check_func,
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        expected_graph_size=expected_graph_size
    )

    assert results['pct_semantically_valid'] == 50.0
    assert results['pct_valid_but_wrong_size'] == 0.0
    assert results['pct_valid_novel_graphs'] == 25.0
    assert results['pct_novel_graphs'] == 50.0
    assert results['pct_original_graphs'] == 25.0
    assert results['pct_empty_graphs'] == 25.0

