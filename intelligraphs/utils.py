def load_strings(file, split_tab=False):
    """
    Load strings from file.

    Args:
        file (str): Path to file.
        split_tab (bool): If True, split strings by tab. Else, split by space.

    Returns:
        list: List of strings.
    """
    with open(file, 'r') as f:
        if split_tab:
            return [line.replace('\n', '').split('\t') for line in f]
        else:
            return [line.split() for line in f]


def split_subgaphs(x):
    """
    Split subgraphs.

    Args:
        x (list): List of strings.

    Returns:
        list: List of subgraphs.
    """
    y = list()
    z = list()
    for i in x:
        if not i == ['']:
            z.append(i)
        else:
            y.append(z)
            z = list()
    return y


def create_mapping(train_file, val_file, test_file):
    """
    Create entity and relation mapping. This mapping is later used to convert the strings to integers.

    Args:
        train_file (str): Path to train file.
        val_file (str): Path to validation file.
        test_file (str): Path to test file.

    Returns:
        tuple: Tuple of entity and relation mappings.
    """
    train = load_strings(train_file, split_tab=True)
    val = load_strings(val_file, split_tab=True)
    test = load_strings(test_file, split_tab=True)

    nodes, rels = set(), set()
    for triple in train + val + test:
        if triple == ['']:  # skip empty lines used to separate subgraphs
            continue
        nodes.add(triple[0])
        rels.add(triple[1])
        nodes.add(triple[2])

    i2n, i2r = list(nodes), list(rels)
    n2i, r2i = {n: i for i, n in enumerate(nodes)}, {r: i for i, r in enumerate(rels)}

    max_graph_len = max([len(x) for x in train + val + test])
    return (n2i, i2n), (r2i, i2r), max_graph_len


def map_nodes_relations(data, n2i, r2i):
    """
    Convert strings to integers.

    Args:
        data (list): List of subgraphs.
        n2i (dict): Node to integer mapping.
        r2i (dict): Relation to integer mapping.

    Returns:
        list: List of subgraphs with integers.
    """
    mapped_data = []
    for subgraph in data:
        x = []
        for s, p, o in subgraph:
            x.append([n2i[s], r2i[p], n2i[o]])
        mapped_data.append(x)
    return mapped_data


def pad_data(data, max_len):
    """
    Pad subgraphs with empty triples.

    Args:
        data (list): List of subgraphs.
        max_len (int): Maximum length of subgraphs.

    Returns:
        list: List of padded subgraphs.
    """

    import warnings
    warnings.warn(f'Padding subgraphs with empty triples. {max_len}')

    # For every subgraph, pad with empty triples if the length is less than max_len
    for i in range(len(data)):
        while len(data[i]) < max_len:
            data[i].append([-1, -1, -1])
    return data
