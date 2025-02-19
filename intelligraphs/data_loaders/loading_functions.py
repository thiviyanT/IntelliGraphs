import torch

from intelligraphs.data_loaders.utils import (
    get_file_paths,
    pad_to_max_length,
    compute_statistics,
    parse_files_to_subgraphs,
    process_knowledge_graphs
)


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
    train_list, valid_list, test_list, entity_mappings, relation_mappings, edge_stats, entity_stats = (
        load_data_as_list(dataset, limit=limit))
    min_edges, max_edges = edge_stats

    # Graphs must be padded to ensure that they are the same lengths
    traini = pad_to_max_length(train_list, max_edges)
    vali = pad_to_max_length(valid_list, max_edges)
    testi = pad_to_max_length(test_list, max_edges)

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
    train_file, val_file, test_file = get_file_paths(dataset)
    train, val, test = parse_files_to_subgraphs(train_file, val_file, test_file, split_tab=True)

    # limit the number of graphs (optional) - helpful for debugging
    if limit:
        train = train[:limit]
        val = val[:limit]
        test = test[:limit]

    # process graphs to get indexed triples and mappings
    train_list, valid_list, test_list, entity_mappings, relation_mappings, max_length = (
        process_knowledge_graphs(train, val, test))

    edge_stats, entity_stats = compute_statistics(train_list, valid_list, test_list)

    return train_list, valid_list, test_list, entity_mappings, relation_mappings, edge_stats, entity_stats
