from typing import List, Tuple, Dict, Any, Union
import torch


def post_process_data(input: Any,
                      entity_id_to_label: Dict[int, str] = None,
                      relation_id_to_label: Dict[int, str] = None) -> List[List[Tuple[str, str, str]]]:
    """
    Process the input by converting tensors to lists (if not already a list), labeling graphs, and performing integer type conversion.

    Parameters:
    - input: Tensor or List of graphs.
    - entity_id_to_label: Dictionary mapping node IDs to node labels.
    - relation_id_to_label: Dictionary mapping relation IDs to relation labels.

    Returns:
    - A list of preprocessed graphs with labeled entities and relations.
    """
    assert isinstance(input, (torch.Tensor, list)), "Input must be a PyTorch tensor or a list."
    assert entity_id_to_label is not None, "Entity ID to label mapping cannot be None."
    assert relation_id_to_label is not None, "Relation ID to label mapping cannot be None."

    # If input is a tensor, convert it to a list of lists
    if isinstance(input, torch.Tensor):
        input_list = input.tolist()
    else:
        input_list = input

    # Remove padding
    unpadded_list = [[triple for triple in subgraph if triple != [-1, -1, -1]] for subgraph in input_list]

    # Label subgraphs using entity_id_to_label and relation_id_to_label dictionaries
    labeled_graphs = label_subgraphs(unpadded_list, entity_id_to_label, relation_id_to_label)

    # Convert string types to int types where applicable
    processed_graphs = int_type_conversion(labeled_graphs)

    return processed_graphs


def label_subgraphs(input_list: List[List[Tuple[int, int, int]]],
                    i2e: Dict[int, str] = None,
                    i2r: Dict[int, str] = None) -> List[List[Tuple[str, str, str]]]:
    """
    Convert a list of graphs with integer IDs to labeled graphs using provided dictionaries.

    Parameters:
    - input_list: List of graphs where each subgraph is represented by (subject_id, relation_id, object_id).
    - i2e: Dictionary mapping node IDs to node labels.
    - i2r: Dictionary mapping relation IDs to relation labels.

    Returns:
    - A list of labeled graphs.
    """
    assert isinstance(input_list, list), "Input must be a list."
    assert i2e is not None, "Entity ID to label dictionary cannot be None."
    assert i2r is not None, "Relation ID to label dictionary cannot be None."

    return [
        [
            (i2e[sub[0]], i2r[sub[1]], i2e[sub[2]])  # Convert IDs to labels using i2e and i2r
            for sub in subgraph
        ]
        for subgraph in input_list
    ]


def int_type_conversion(graphs: List[List[Tuple[str, str, str]]] = []) -> List[List[List[Union[int, str]]]]:
    """
    Converts string types to integer types where applicable.

    Parameters:
    - graphs: The list of graphs represented as lists of triples (subject, relation, object).

    Returns:
    - A list of graphs where string types are converted to integers where applicable.
    """
    assert isinstance(graphs, list), "Graphs must be a list of lists of triples."
    assert len(graphs) > 1, "Graph list cannot be empty"

    return [
        [
            [int(element) if element.isdigit() else element for element in triple]
            for triple in graph
        ]
        for graph in graphs
    ]