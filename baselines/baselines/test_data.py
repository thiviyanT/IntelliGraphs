from data import compute_min_max_edges, compute_min_max_nodes


def test_compute_min_max_edges():
    # Test cases with expected results
    test_cases = [
        # Single subgraph with 3 edges
            # Subgraph 1: 3 edges
        ([[[1, 0, 2], [2, 1, 3], [3, 2, 4]]], (3, 3)),

        # Multiple subgraphs with varying number of edges
            # Subgraph 1: 1 edge
            # Subgraph 2: 2 edges
            # Subgraph 3: 3 edges
        ([[[1, 0, 2]], [[2, 1, 3], [3, 2, 4]], [[4, 3, 5], [5, 4, 6], [6, 5, 7]]], (1, 3)),

        # Empty subgraph
            # Subgraph 1: 0 edges -> [[]]
        ([[]], (0, 0)),
        # Multiple subgraphs with an empty subgraph
            # Subgraph 1: 1 edge
            # Subgraph 2: 0 edges
            # Subgraph 3: 1 edge
        ([[[1, 0, 2]], [], [[3, 1, 4]]], (0, 1)),
    ]

    for subgraphs, expected in test_cases:
        result = compute_min_max_edges(subgraphs)
        assert result == expected, f"Failed for {subgraphs}. Expected {expected}, got {result}"
    print("All tests for compute_min_max_edges passed!")


def test_compute_min_max_nodes():
    # Test cases with expected results
    test_cases = [
        # Single subgraph with 4 nodes: {1, 2, 3, 4}
        ([
             [[1, 0, 2], [2, 1, 3], [3, 2, 4]]
         ], (4, 4)),
        # Multiple subgraphs:
            # First subgraph has 2 nodes: {1, 2}
            # Second subgraph has 3 nodes: {2, 3, 4}
            # Third subgraph has 4 nodes: {4, 5, 6, 7}
        ([
             [[1, 0, 2]],
             [[2, 1, 3], [3, 2, 4]],
             [[4, 3, 5], [5, 4, 6], [6, 5, 7]]
         ], (2, 4)),
        # Empty subgraph
        ([[]], (0, 0)),
        # Multiple subgraphs with an empty subgraph:
            # First subgraph has 2 nodes: {1, 2}
            # Second subgraph is empty: {}
            # Third subgraph has 2 nodes: {3, 4}
        ([
             [[1, 0, 2]],
             [],
             [[3, 1, 4]]
         ], (0, 2)),
    ]

    for subgraphs, expected in test_cases:
        result = compute_min_max_nodes(subgraphs)
        assert result == expected, f"Failed for {subgraphs}. Expected {expected}, got {result}"
    print("All tests for compute_min_max_nodes passed!")


# Running the tests
test_compute_min_max_edges()
test_compute_min_max_nodes()
