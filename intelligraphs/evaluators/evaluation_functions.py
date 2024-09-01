from typing import List, Tuple, Dict, Callable



def check_semantics(
        sampled_graphs: List[List[Tuple[str, str, str]]],
        training_data: List[List[Tuple[str, str, str]]],
        semantic_check_func: Callable,
        entity_labels: Dict[int, str] = None,
        relation_labels: Dict[int, str] = None,
        expected_graph_size: int = None
) -> Tuple[Dict[str, float], Dict[str, List[List[Tuple[str, str, str]]]]]:
    """
    Main function to check the semantic validity of sampled graphs and categorize them based on properties.

    :param sampled_graphs: A list of sampled graphs, where each graph is a list of triples (subject, predicate, object).
    :param training_data: A list of graphs from the training data for comparison to determine novelty.
    :param semantic_check_func: A function that checks the semantic validity of a graph.
    :param entity_labels: A dictionary mapping entity indices to their names (optional).
    :param relation_labels: A dictionary mapping relation indices to their names (optional).
    :param expected_graph_size: The expected size of a valid graph (optional).
    :return: A tuple containing two dictionaries:
             - The first dictionary contains percentages of different graph properties.
             - The second dictionary contains lists of graphs categorized by their properties.
    """

    valid_graphs = []
    valid_novel_graphs = []
    invalid_graphs = []

    empty_graph_count, original_graph_count, novel_graph_count = 0, 0, 0
    valid_graph_count, valid_novel_graph_count, valid_but_wrong_size_count = 0, 0, 0

    for graph in sampled_graphs:
        if is_graph_empty(graph):
            empty_graph_count += 1
            continue

        if is_graph_in_training_data(graph, training_data):
            original_graph_count += 1
            valid, valid_but_wrong_size = validate_graph(graph, semantic_check_func, entity_labels, relation_labels,
                                                         expected_graph_size)
            if valid:
                valid_graph_count += 1
                valid_graphs.append(graph)
            elif valid_but_wrong_size:
                valid_but_wrong_size_count += 1
                valid_graphs.append(graph)
            else:
                invalid_graphs.append(graph)
        else:
            novel_graph_count += 1
            valid, valid_but_wrong_size = validate_graph(graph, semantic_check_func, entity_labels, relation_labels,
                                                         expected_graph_size)
            if valid:
                valid_graph_count += 1
                valid_novel_graph_count += 1
                valid_novel_graphs.append(graph)
            elif valid_but_wrong_size:
                valid_but_wrong_size_count += 1
                valid_novel_graphs.append(graph)
            else:
                invalid_graphs.append(graph)

    return compile_results(
        len(sampled_graphs),
        valid_graph_count,
        valid_novel_graph_count,
        valid_but_wrong_size_count,
        novel_graph_count,
        original_graph_count,
        empty_graph_count,
        valid_graphs,
        valid_novel_graphs,
        invalid_graphs
    )


def is_graph_empty(graph: List[Tuple[str, str, str]]) -> bool:
    """Check if the graph is empty."""
    return len(graph) == 0


def is_graph_in_training_data(graph: List[Tuple[str, str, str]],
                              training_data: List[List[Tuple[str, str, str]]]) -> bool:
    """Check if the graph is present in the training data."""
    return graph in training_data


def validate_graph(
        graph: List[Tuple[str, str, str]],
        semantic_check_func: Callable,
        entity_labels: Dict[int, str],
        relation_labels: Dict[int, str],
        expected_graph_size: int
) -> Tuple[bool, bool]:
    """
    Validate the graph for semantic correctness and expected size.

    :param graph: The graph to validate.
    :param semantic_check_func: A function that checks the semantic validity of a graph.
    :param entity_labels: A dictionary mapping entity indices to their names.
    :param relation_labels: A dictionary mapping relation indices to their names.
    :param expected_graph_size: The expected size of a valid graph.
    :return: A tuple (valid, valid_but_wrong_size) where:
             - valid is True if the graph is semantically valid and of the expected size.
             - valid_but_wrong_size is True if the graph is semantically valid but of a different size.
    """
    if semantic_check_func(graph, entity_labels, relation_labels, length=expected_graph_size):
        return True, False
    elif semantic_check_func(graph, entity_labels, relation_labels, length=None):
        return False, True
    return False, False


def compile_results(
        total_graphs: int,
        valid_graph_count: int,
        valid_novel_graph_count: int,
        valid_but_wrong_size_count: int,
        novel_graph_count: int,
        original_graph_count: int,
        empty_graph_count: int,
        valid_graphs: List[List[Tuple[str, str, str]]],
        valid_novel_graphs: List[List[Tuple[str, str, str]]],
        invalid_graphs: List[List[Tuple[str, str, str]]]
) -> Tuple[Dict[str, float], Dict[str, List[List[Tuple[str, str, str]]]]]:
    """
    Compile and calculate the results for reporting.

    :param total_graphs: The total number of sampled graphs.
    :param valid_graph_count: The number of valid graphs.
    :param valid_novel_graph_count: The number of valid novel graphs.
    :param valid_but_wrong_size_count: The number of valid graphs with the wrong size.
    :param novel_graph_count: The number of novel graphs.
    :param original_graph_count: The number of graphs found in the training data.
    :param empty_graph_count: The number of empty graphs.
    :param valid_graphs: The list of valid graphs.
    :param valid_novel_graphs: The list of valid novel graphs.
    :param invalid_graphs: The list of invalid graphs.
    :return: A tuple containing two dictionaries:
             - The first dictionary contains percentages of different graph properties.
             - The second dictionary contains lists of graphs categorized by their properties.
    """
    pct_semantically_valid = round((valid_graph_count / total_graphs) * 100, 2)
    pct_valid_novel_graphs = round((valid_novel_graph_count / total_graphs) * 100, 2)
    pct_novel_graphs = round((novel_graph_count / total_graphs) * 100, 2)
    pct_original_graphs = round((original_graph_count / total_graphs) * 100, 2)
    pct_empty_graphs = round((empty_graph_count / total_graphs) * 100, 2)
    pct_valid_but_wrong_size = round((valid_but_wrong_size_count / total_graphs) * 100, 2)

    results = {
        'pct_semantically_valid': pct_semantically_valid,
        'pct_valid_but_wrong_size': pct_valid_but_wrong_size,
        'pct_valid_novel_graphs': pct_valid_novel_graphs,
        'pct_novel_graphs': pct_novel_graphs,
        'pct_original_graphs': pct_original_graphs,
        'pct_empty_graphs': pct_empty_graphs
    }

    categorized_graphs = {
        'valid_graphs': valid_graphs,
        'valid_novel_graphs': valid_novel_graphs,
        'invalid_graphs': invalid_graphs
    }

    return results, categorized_graphs

