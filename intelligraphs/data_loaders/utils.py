import torch
import os


def load_strings(file, split_tab=False):
    """
    Load and process lines from a file.

    :param file: Path to the file containing strings.
    :param split_tab: Boolean flag indicating whether to split each line by tabs ('\t').
                      If False, lines are split by whitespace. Default is False.
    :return: List of lists, where each inner list represents a processed line from the file.
    """
    with open(file, 'r') as f:
        if split_tab:
            return [line.replace('\n', '').split('\t') for line in f]
        else:
            return [line.split() for line in f]


def split_subgraphs(lists):
    """
    Split a list of lists into subgraphs based on empty lists as separators.

    :param lists: List of lists, where each inner list contains strings. Empty lists are used as separators.
    :return: A list of subgraphs, where each subgraph is a list of lists of strings.
    """
    subgraphs = []
    current_subgraph = []

    for inner_list in lists:
        if inner_list != ['']:
            current_subgraph.append(inner_list)
        else:
            subgraphs.append(current_subgraph)
            current_subgraph = []

    return subgraphs


def compute_min_max_edges(subgraphs):
    """
    Compute the minimum and maximum number of edges in subgraphs.

    :param subgraphs: List of subgraphs, where each subgraph is a list of triples.
    :return: A tuple containing (min_edges, max_edges)
    """
    min_edges, max_edges = float('inf'), 0

    for subgraph in subgraphs:
        num_edges = len(subgraph)
        min_edges = min(min_edges, num_edges)
        max_edges = max(max_edges, num_edges)

    return min_edges, max_edges


def compute_min_max_entities(subgraphs):
    """
    Compute the minimum and maximum number of nodes in subgraphs.

    :param subgraphs: List of subgraphs, where each subgraph is a list of triples.
    :return: A tuple containing (min_nodes, max_nodes)
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


def load_data_files(dataset):
    """
    Loads the train, validation, and test data from TSV files.

    Args:
        dataset (str): The dataset of the dataset directory.

    Returns:
        tuple: Three lists containing raw subgraphs for train, validation, and test splits.
    """
    dir = '.data' + os.sep + dataset
    if os.path.isdir(dir):
        train_file = f'.data/{dataset}/train_split.tsv'
        val_file = f'.data/{dataset}/val_split.tsv'
        test_file = f'.data/{dataset}/test_split.tsv'
    else:
        raise Exception(f'Could not find dataset with filename {dataset} at location "{dir}".')

    train = load_strings(train_file, split_tab=True)
    val = load_strings(val_file, split_tab=True)
    test = load_strings(test_file, split_tab=True)

    return train, val, test


def process_subgraphs(train, val, test):
    """
    Processes subgraphs to convert them into indexed triples and creates mappings for entities and relations.

    Args:
        train, val, test (list): Lists of subgraphs for train, validation, and test splits.

    Returns:
        tuple: Indexed subgraphs, entity mappings, and relation mappings.
    """
    entities, relations = set(), set()
    for subgraph in train + val + test:
        for triple in subgraph:
            subject, predicate, obj = triple
            entities.add(subject)
            relations.add(predicate)
            entities.add(obj)

    e2i = {n: i for i, n in enumerate(entities)}
    r2i = {r: i for i, r in enumerate(relations)}
    i2e = {i: n for i, n in enumerate(entities)}
    i2r = {i: r for i, r in enumerate(relations)}

    def index_subgraphs(subgraphs):
        indexed = []
        for subgraph in subgraphs:
            x = []
            for s, p, o in subgraph:
                x.append([e2i[s], r2i[p], e2i[o]])
            indexed.append(x)
        return indexed

    traini = index_subgraphs(train)
    vali = index_subgraphs(val)
    testi = index_subgraphs(test)

    return traini, vali, testi, (e2i, i2e), (r2i, i2r)


def pad_subgraphs(subgraphs, max_edges):
    """
    Pads subgraphs with [-1, -1, -1] to ensure uniform length.

    Args:
        subgraphs (list): List of indexed subgraphs.
        max_edges (int): Maximum number of edges in any subgraph.

    Returns:
        list: Padded subgraphs.
    """
    for i in range(len(subgraphs)):
        while len(subgraphs[i]) < max_edges:
            subgraphs[i].append([-1, -1, -1])
    return subgraphs


def compute_statistics(train, val, test):
    """
    Computes the min/max number of entities and edges in the subgraphs.

    Args:
        train, val, test (list): Lists of subgraphs for train, validation, and test splits.

    Returns:
        tuple: Min/max edges and min/max entities.
    """
    min_entities, max_entities = compute_min_max_entities(train + val + test)
    min_edges, max_edges = compute_min_max_edges(train + val + test)
    return (min_edges, max_edges), (min_entities, max_entities)


def load_data_as_tensor(dataset, limit=None):
    """
    Loads and processes a dataset from the specified directory, returning the data as PyTorch tensors.

    Args:
        dataset (str): The name of the dataset directory containing the knowledge graphs.
        limit (int, optional): If provided, limits the number of subgraphs loaded from each split (train, validation, test).

    Returns:
        tuple: A tuple containing:
            - train_tensor (torch.Tensor): The processed and padded training subgraphs as a PyTorch tensor.
            - val_tensor (torch.Tensor): The processed and padded validation subgraphs as a PyTorch tensor.
            - test_tensor (torch.Tensor): The processed and padded test subgraphs as a PyTorch tensor.
            - entity_mappings (tuple): A tuple containing two dictionaries:
                - e2i (dict): A mapping from entity names to indices.
                - i2e (dict): A mapping from indices to entity names.
            - relation_mappings (tuple): A tuple containing two dictionaries:
                - r2i (dict): A mapping from relation names to indices.
                - i2r (dict): A mapping from indices to relation names.
            - edge_stats (tuple): A tuple containing:
                - min_edges (int): The minimum number of edges in any subgraph.
                - max_edges (int): The maximum number of edges in any subgraph.
            - entity_stats (tuple): A tuple containing:
                - min_entities (int): The minimum number of entities in any subgraph.
                - max_entities (int): The maximum number of entities in any subgraph.
    """
    train_list, valid_list, test_list, entity_mappings, relation_mappings, edge_stats, entity_stats = load_data_as_list(dataset, limit=limit)
    min_edges, max_edges = edge_stats

    # Graphs must be padded to ensure that they are the same lengths
    traini = pad_subgraphs(train_list, max_edges)
    vali = pad_subgraphs(valid_list, max_edges)
    testi = pad_subgraphs(test_list, max_edges)

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(traini)
    val_tensor = torch.tensor(vali)
    test_tensor = torch.tensor(testi)

    return train_tensor, val_tensor, test_tensor, entity_mappings, relation_mappings, edge_stats, entity_stats


def load_data_as_list(dataset, limit=None):
    """
    Loads and processes a dataset from the specified directory, returning the data as lists.

    Args:
        dataset (str): The name of the dataset directory containing the knowledge graphs.
        limit (int, optional): If provided, limits the number of subgraphs loaded from each split (train, validation, test).

    Returns:
        tuple: A tuple containing:
            - train_list (list): The processed training subgraphs as a list of indexed triples.
            - valid_list (list): The processed validation subgraphs as a list of indexed triples.
            - test_list (list): The processed test subgraphs as a list of indexed triples.
            - entity_mappings (tuple): A tuple containing two dictionaries:
                - e2i (dict): A mapping from entity names to indices.
                - i2e (dict): A mapping from indices to entity names.
            - relation_mappings (tuple): A tuple containing two dictionaries:
                - r2i (dict): A mapping from relation names to indices.
                - i2r (dict): A mapping from indices to relation names.
            - edge_stats (tuple): A tuple containing:
                - min_edges (int): The minimum number of edges in any subgraph.
                - max_edges (int): The maximum number of edges in any subgraph.
            - entity_stats (tuple): A tuple containing:
                - min_entities (int): The minimum number of entities in any subgraph.
                - max_entities (int): The maximum number of entities in any subgraph.
    """
    train, val, test = load_data_files(dataset)

    # Split into structured subgraphs
    train = split_subgraphs(train)
    val = split_subgraphs(val)
    test = split_subgraphs(test)

    # (Optionally) limit the number of graphs
    if limit:
        train = train[:limit]
        val = val[:limit]
        test = test[:limit]

    # Process graphs to get indexed triples and mappings
    train_list, valid_list, test_list, entity_mappings, relation_mappings = process_subgraphs(train, val, test)

    # Compute statistics on the dataset
    edge_stats, entity_stats = compute_statistics(train_list, valid_list, test_list)

    return train_list, valid_list, test_list, entity_mappings, relation_mappings, edge_stats, entity_stats
