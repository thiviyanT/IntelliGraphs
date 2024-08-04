import torch
import os
import util

VALPROP = 0.4
REST = '.rest'
INV = '.inv'

S = os.sep


def add_neighbors(set, graph, node, depth=2):
    if depth == 0:
        return

    for s, p, o in graph.triples((node, None, None)):
        set.add((s, p, o))
        add_neighbors(set, graph, o, depth=depth - 1)

    for s, p, o in graph.triples((None, None, node)):
        set.add((s, p, o))
        add_neighbors(set, graph, s, depth=depth - 1)


def load_strings(file, split_tab=False):
    with open(file, 'r') as f:
        if split_tab:
            return [line.replace('\n', '').split('\t') for line in f]
        else:
            return [line.split() for line in f]


def split_subgaphs(x):
    y = list()
    z = list()
    for i in x:
        if not i == ['']:
            z.append(i)
        else:
            y.append(z)
            z = list()
    return y


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


def compute_min_max_nodes(subgraphs):
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


def load_data(name, limit=None, padding=False):
    """
    :return: Three lists of integer-triples (train, val, test), a pair of dicts to map entity strings from an entity
        to their integer ids, and a similar pair of dicts for the relations.
    """

    dir = '.data' + os.sep + name
    if os.path.isdir(dir):
        train_file = f'.data/{name}/train.txt'
        val_file = f'.data/{name}/valid.txt'
        test_file = f'.data/{name}/test.txt'
    else:
        raise Exception(f'Could not find dataset with name {name} at location "{dir}".')

    train = load_strings(train_file, split_tab=True)
    val = load_strings(val_file, split_tab=True)
    test = load_strings(test_file, split_tab=True)

    train = split_subgaphs(train)
    val = split_subgaphs(val)
    test = split_subgaphs(test)

    if limit:
        train = train[:limit]
        val = val[:limit]
        test = test[:limit]

    # mappings for nodes (n) and relations (r)
    nodes, rels = set(), set()
    for subgraph in train + val + test:
        for triple in subgraph:
            nodes.add(triple[0])
            rels.add(triple[1])
            nodes.add(triple[2])

    i2n, i2r = list(nodes), list(rels)
    n2i, r2i = {n: i for i, n in enumerate(nodes)}, {r: i for i, r in enumerate(rels)}

    traini, vali, testi = [], [], []

    for subgraph in train:
        x = []
        for s, p, o in subgraph:
            x.append([n2i[s], r2i[p], n2i[o]])
        traini.append(x)

    for subgraph in val:
        x = []
        for s, p, o in subgraph:
            x.append([n2i[s], r2i[p], n2i[o]])
        vali.append(x)

    for subgraph in test:
        x = []
        for s, p, o in subgraph:
            x.append([n2i[s], r2i[p], n2i[o]])
        testi.append(x)

    # Pad subgraphs with [-1,-1,-1] to make them equal length
    if padding:
        # Add a warning here so that user is aware
        import warnings
        warnings.warn('Padding subgraphs with empty triples')

        # Extract the min/max number
        min_nodes, max_nodes = compute_min_max_nodes(train + val + test)

        # Extract the min/max number of edges
        min_edges, max_edges = compute_min_max_edges(train + val + test)

        for i in range(len(traini)):
            while len(traini[i]) < max_edges:
                traini[i].append([-1, -1, -1])
        for i in range(len(vali)):
            while len(vali[i]) < max_edges:
                vali[i].append([-1, -1, -1])
        for i in range(len(testi)):
            while len(testi[i]) < max_edges:
                testi[i].append([-1, -1, -1])

    train, val, test = torch.tensor(traini), torch.tensor(vali), torch.tensor(testi)
    return train, val, test, (n2i, i2n), (r2i, i2r), (min_edges, max_edges), (min_nodes, max_nodes)
