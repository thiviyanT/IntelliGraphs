import pytest
from intelligraphs.evaluators import SemanticEvaluator

# Mock data for testing
mock_predicted_graphs = [
    ['A', 'B', 'C'],
    ['X', 'Y'],
    [],
    ['D', 'E', 'F'],
    ['X', 'Y'],  # Duplicate to test novel/original detection
]

mock_ground_truth_graphs = [
    ['A', 'B', 'C'],
    ['D', 'E', 'F'],
]

def mock_semantic_func(graph, entity_labels, relation_labels):
    # A mock function that returns True if the graph contains more than one element
    return len(graph) > 1

def test_initialization():
    checker = SemanticEvaluator(mock_predicted_graphs, mock_ground_truth_graphs, mock_semantic_func)

    assert isinstance(checker.predicted_graphs, list)
    assert isinstance(checker.ground_truth_graphs, list)
    assert callable(checker.func)
    assert checker.valid == 0
    assert isinstance(checker.valid_graphs, set)
    assert isinstance(checker.valid_novel_graphs, set)
    assert isinstance(checker.invalid_graphs, set)
    assert checker.empty_graph == 0
    assert checker.novel_graphs == 0
    assert checker.original_graphs == 0

def test_is_empty_graph():
    checker = SemanticEvaluator(mock_predicted_graphs, mock_ground_truth_graphs, mock_semantic_func)

    assert checker.is_empty_graph([]) is True
    assert checker.is_empty_graph(['A']) is False

    with pytest.raises(AssertionError):
        checker.is_empty_graph("Not a list or tuple")

def test_check_novelty():
    checker = SemanticEvaluator(mock_predicted_graphs, mock_ground_truth_graphs, mock_semantic_func)

    assert checker.check_novelty(['X', 'Y']) is True
    assert checker.check_novelty(['A', 'B', 'C']) is False

    with pytest.raises(AssertionError):
        checker.check_novelty("Not a list or tuple")

def test_check_graph():
    checker = SemanticEvaluator(mock_predicted_graphs, mock_ground_truth_graphs, mock_semantic_func)

    # Test an empty graph
    assert checker.check_graph([]) is False
    assert checker.empty_graph == 1

    # Test an original valid graph
    assert checker.check_graph(['A', 'B', 'C']) is True
    assert checker.valid == 1
    assert checker.original_graphs == 1
    assert 'A' in next(iter(checker.valid_graphs))

    # Test a novel valid graph
    assert checker.check_graph(['X', 'Y']) is True
    assert checker.valid == 2
    assert checker.novel_graphs == 1
    assert 'X' in next(iter(checker.valid_novel_graphs))

    # Test an invalid graph
    assert checker.check_graph(['Invalid']) is False
    assert 'Invalid' in next(iter(checker.invalid_graphs))

def test_evaluate_graphs():
    checker = SemanticEvaluator(mock_predicted_graphs, mock_ground_truth_graphs, mock_semantic_func)
    results = checker.evaluate_graphs()

    assert results['results']['semantics'] == 80.0  # 4 out of 5 graphs are valid
    assert results['results']['novel_semantics'] == 20.0  # 1 novel valid graph
    assert results['results']['novel'] == 20.0  # 1 novel graph out of 5
    assert results['results']['original'] == 40.0  # 2 original graphs out of 5
    assert results['results']['empty'] == 20.0  # 1 empty graph out of 5

def test_print_results(capsys):
    checker = SemanticEvaluator(mock_predicted_graphs, mock_ground_truth_graphs, mock_semantic_func)
    checker.evaluate_graphs()
    checker.print_results()

    captured = capsys.readouterr()
    assert "Percentage Results:" in captured.out
    assert "Graph Results:" in captured.out
    assert "Semantics: 80.0%" in captured.out
    assert "Count: 2" in captured.out  # For valid_graphs or another graph set
