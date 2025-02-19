from typing import List, Dict, Tuple, Union
import os


def get_file_paths(dataset_name, data_dir='.data') -> Tuple[str, str, str]:
    """Get paths for train, validation and test files."""
    dataset_folder_path = os.path.join(data_dir, dataset_name)
    return (
        os.path.join(dataset_folder_path, 'train_split.tsv'),
        os.path.join(dataset_folder_path, 'val_split.tsv'),
        os.path.join(dataset_folder_path, 'test_split.tsv')
    )


def read_tsv_file(file: str, split_tab: bool = True) -> List[List[str]]:
    """
    Load triples from TSV files.

    Args:
        file (str): Path to the input file.
        split_tab (bool): If True, elements in the triplets are separated using tabs as deliminator. If False, splits by spaces.

    Returns:
        List[List[str]]: A list where each inner list contains the tokens from one line.
    """
    with open(file, 'r') as f:
        if split_tab:
            return [line.replace('\n', '').split('\t') for line in f]
        else:
            return [line.split() for line in f]


def convert_to_indices(
    data: List[List[List[str]]], 
    e2i: Dict[str, int], 
    r2i: Dict[str, int]
) -> List[List[List[int]]]:
    """
    Convert strings to integers.

    Args:
        data (list): List of subgraphs.
        e2i (dict): Node to integer mapping.
        r2i (dict): Relation to integer mapping.

    Returns:
        list: List of subgraphs with integers.
    """
    mapped_data = []
    for subgraph in data:
        x = []
        for s, p, o in subgraph:
            x.append([e2i[s], r2i[p], e2i[o]])
        mapped_data.append(x)
    return mapped_data


def pad_to_max_length(
    subgraphs: List[List[List[int]]], 
    max_edges: int
) -> List[List[List[int]]]:
    """
    Pads subgraphs with [-1, -1, -1] to ensure uniform length.

    Args:
        subgraphs (list): List of indexed subgraphs.
        max_edges (int): Maximum number of edges in any subgraph.

    Returns:
        list: Padded subgraphs.
    """
    # For every subgraph, pad with empty triples if the length is less than max_len
    for i in range(len(subgraphs)):
        while len(subgraphs[i]) < max_edges:
            subgraphs[i].append([-1, -1, -1])
    return subgraphs


def parse_subgraphs(lists: List[List[str]]) -> List[List[List[str]]]:
    """
    Split a list of lists into subgraphs based on empty lists as separators.

    Args:
        lists: List of lists, where each inner list contains strings. Empty lists are used as separators.
    Returns:
        A list of subgraphs, where each subgraph is a list of lists of strings.
    """
    subgraphs = []
    current_subgraph = []

    for inner_list in lists:
        if inner_list != ['']:
            current_subgraph.append(inner_list)
        else:
            if current_subgraph:  # Only append non-empty subgraphs
                subgraphs.append(current_subgraph)
            current_subgraph = []

    # Add the last subgraph if it's not empty
    if current_subgraph:
        subgraphs.append(current_subgraph)

    return subgraphs


def parse_files_to_subgraphs(
        train_file: str,
        val_file: str,
        test_file: str,
        split_tab: bool = True
) -> Tuple[List[List[List[str]]], List[List[List[str]]], List[List[List[str]]]]:
    """
    Read and parse files into subgraphs. This is the first step in the processing pipeline.

    Args:
        train_file (str): Path to train file.
        val_file (str): Path to validation file.
        test_file (str): Path to test file.
        split_tab (bool): If True, split strings by tab. Else, split by space.

    Returns:
        tuple: Tuple containing (train_subgraphs, val_subgraphs, test_subgraphs).
    """
    train_raw = read_tsv_file(train_file, split_tab=split_tab)
    val_raw = read_tsv_file(val_file, split_tab=split_tab)
    test_raw = read_tsv_file(test_file, split_tab=split_tab)

    # Parse into subgraphs
    train_subgraphs = parse_subgraphs(train_raw)
    val_subgraphs = parse_subgraphs(val_raw)
    test_subgraphs = parse_subgraphs(test_raw)

    return train_subgraphs, val_subgraphs, test_subgraphs


def create_mappings_from_subgraphs(
        subgraphs: List[List[List[str]]]
    ) -> Tuple[Tuple[Dict[str, int], Dict[int, str]],
               Tuple[Dict[str, int], Dict[int, str]]]:
    """
    Create entity and relation mappings from subgraphs. This function handles the conversion
    of strings to indices.

    Args:
        subgraphs (list): List of all subgraphs to create mappings from.

    Returns:
        tuple: ((entity_to_idx, idx_to_entity), (relation_to_idx, idx_to_relation))
    """
    entities, relations = set(), set()
    for subgraph in subgraphs:
        for subject, predicate, obj in subgraph:
            entities.add(subject)
            relations.add(predicate)
            entities.add(obj)

    # bidirectional mappings
    e2i = {n: i for i, n in enumerate(entities)}
    r2i = {r: i for i, r in enumerate(relations)}
    i2e = {i: n for n, i in e2i.items()}
    i2r = {i: r for r, i in r2i.items()}

    return (e2i, i2e), (r2i, i2r)


def process_knowledge_graphs(
        train_file: str,
        val_file: str,
        test_file: str
    ) -> Tuple[
        List[List[List[int]]],
        List[List[List[int]]],
        List[List[List[int]]],
        Tuple[Dict[str, int], Dict[int, str]],
        Tuple[Dict[str, int], Dict[int, str]],
        int
    ]:
    """
    Complete pipeline for processing knowledge graph data. This function combines parsing,
    mapping creation, and conversion to indices.

    Args:
        train_file (str): Path to train file.
        val_file (str): Path to validation file.
        test_file (str): Path to test file.

    Returns:
        tuple: (indexed_train, indexed_val, indexed_test, entity_mappings, relation_mappings, max_graph_len)
    """
    train_graphs, val_graphs, test_graphs = parse_files_to_subgraphs(
        train_file, val_file, test_file
    )

    all_graphs = train_graphs + val_graphs + test_graphs
    entity_mappings, relation_mappings = create_mappings_from_subgraphs(all_graphs)
    e2i, _ = entity_mappings
    r2i, _ = relation_mappings

    train_indexed = convert_to_indices(train_graphs, e2i, r2i)
    val_indexed = convert_to_indices(val_graphs, e2i, r2i)
    test_indexed = convert_to_indices(test_graphs, e2i, r2i)

    max_graph_length = max(len(g) for g in all_graphs)

    return (
        train_indexed,
        val_indexed,
        test_indexed,
        entity_mappings,
        relation_mappings,
        max_graph_length
    )


def compute_min_max_edges(
        subgraphs: List[List[List[Union[str, int]]]]
    ) -> Tuple[int, int]:
    """
    Compute the minimum and maximum number of edges in subgraphs.

    Args:
        subgraphs (List[List[List[Union[str, int]]]]): List of subgraphs, where each
            subgraph is a list of triples. Each triple contains subject, predicate,
            and object elements.

    Returns:
        Tuple[int, int]: A tuple containing (min_edges, max_edges), where:
            - min_edges: Minimum number of edges in any subgraph
            - max_edges: Maximum number of edges in any subgraph
    """
    min_edges, max_edges = float('inf'), 0

    for subgraph in subgraphs:
        num_edges = len(subgraph)
        min_edges = min(min_edges, num_edges)
        max_edges = max(max_edges, num_edges)

    return min_edges, max_edges


def compute_min_max_entities(
        subgraphs: List[List[List[Union[str, int]]]]
    ) -> Tuple[int, int]:
    """
    Compute the minimum and maximum number of nodes (entities) in subgraphs.

    Args:
        subgraphs (List[List[List[Union[str, int]]]]): List of subgraphs, where each
            subgraph is a list of triples. Each triple contains subject, predicate,
            and object elements.

    Returns:
        Tuple[int, int]: A tuple containing (min_nodes, max_nodes), where:
            - min_nodes: Minimum number of unique entities in any subgraph
            - max_nodes: Maximum number of unique entities in any subgraph
    """
    min_nodes, max_nodes = float('inf'), 0

    for subgraph in subgraphs:
        nodes = set()
        for s, p, o in subgraph:
            nodes.add(s)
            nodes.add(o)
        num_nodes = len(nodes)
        min_nodes = min(min_nodes, num_nodes)
        max_nodes = max(max_nodes, num_nodes)

    return min_nodes, max_nodes

def compute_statistics(train, val, test):
    """
    Compute statistical information about the subgraphs across all data splits.

    Args:
        train (List[List[List[str]]]): List of training set subgraphs.
        val (List[List[List[str]]]): List of validation set subgraphs.
        test (List[List[List[str]]]): List of test set subgraphs.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: A tuple containing two pairs of statistics:
            - First tuple: (min_edges, max_edges) across all subgraphs
            - Second tuple: (min_entities, max_entities) across all subgraphs
    """
    min_entities, max_entities = compute_min_max_entities(train + val + test)
    min_edges, max_edges = compute_min_max_edges(train + val + test)
    return (min_edges, max_edges), (min_entities, max_entities)
