from _context import models
from scipy.special import gammaln
from tqdm import trange
import torch
import math


def log2comb(n, k):
    """
    Sampling edges from a selection of all possible edges without replacement

    Using gammaln helps to avoid overflow when n is very large

    This is functionally equivalent to doing the following operation:
        bits_p_s_given_e = math.log2( math.comb(n, k)  )

    :param n:
    :param k:
    :return:
    """
    return (gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)) / math.log(2)


def compute_bits_for_synthetic_data(dataset, verbose=False, include_null_entity=False):
    """ Estimate the compression bits for storing graphs sampled using uniform distribution """

    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = models.load_data(dataset, padding=True)

    num_entities = len(e2i)
    # For the IntelliGraphs baseline, we can ignore it because padding is not used
    if include_null_entity:
        num_entities += 1

    num_relations = len(i2r)
    num_edges = train.size(1)

    # Here is we assume the all the triples are similar to the first triple
    subjects = train[0, :, 0]
    objects = train[0, :, 2]
    concatenated = torch.cat((subjects, objects))
    unique_nodes = torch.unique(concatenated)
    num_nodes = len(unique_nodes)

    if verbose:
        print(len(e2i), 'entities')
        print(len(i2r), 'relations')
        print(train.size(0), 'training triples')
        print(test.size(0), 'test triples')
        print(train.size(0) + test.size(0), 'total triples')
        print(num_edges, 'edges')
        print(num_nodes, 'nodes')

    # Model assumes that self-loops are not allowed and edge independence
    num_possible_edges = ((num_nodes)**2-num_nodes) * num_relations
    bits_p_s_given_e = log2comb(num_possible_edges, num_edges)

    # bits_p_e = math.log2(num_entities) * num_nodes
    bits_p_e = log2comb(num_entities, num_nodes)
    compression_bits = bits_p_s_given_e + bits_p_e

    print(f"Dataset: {dataset}")
    print(f"\tNumber of bits for p(S|E):", round(bits_p_s_given_e, 2))
    print(f"\tNumber of bits for p(E):", round(bits_p_e, 2))
    print(f"\tTotal number of bits:", round(bits_p_e + bits_p_s_given_e, 2))
    print(f"\n\n")

    return bits_p_s_given_e, bits_p_e, compression_bits


def compute_bits_for_wikidata_data(dataset, verbose=False):
    """ Estimate the compression bits for storing graphs sampled using uniform distribution """

    train, val, test, (e2i, i2e), (r2i, i2r), (min_edges, max_edges), (min_nodes, max_nodes) = (
        models.load_data(dataset, padding=True))  # TODO: Look into padding

    data = test

    # TODO: For the IntelliGraphs baseline, we can ignore it
    # TODO: For the VAE paper, we should include it
    num_entities = len(e2i) + 1  # Add one for null entities
    num_relations = len(i2r)
    num_graphs = len(data)

    if verbose:
        print(len(e2i), 'entities')
        print(len(i2r), 'relations')
        print(data.size(0), 'training triples')
        print(test.size(0), 'test triples')
        print(data.size(0) + test.size(0), 'total triples')

    bits_p_e, bits_p_s_given_e, compression_bits = 0, 0, 0

    # For every graph compute the compression bits - number of nodes and edges would vary between graph
    for i in trange(len(data)):
        padded_graph = data[i]
        _bits_p_e, _bits_p_s_given_e, _compression_bits = 0, 0, 0

        # Filter out rows that are not equal to the padding triple
        padding_triple = torch.tensor([-1, -1, -1])
        graph = padded_graph[~torch.all(padded_graph == padding_triple, dim=1)]

        subjects = graph[:, 0]
        objects = graph[:, 2]
        concatenated = torch.cat((subjects, objects))
        unique_nodes = torch.unique(concatenated)
        num_nodes = len(unique_nodes)
        num_edges = graph.size(0)

        _bits_p_e += math.log2(max_nodes)  # Explain in the paper that for simplicity we take the maximum nodes from the data (train, val, test)
        _bits_p_e += log2comb(num_entities, num_nodes)

        _bits_p_s_given_e += math.log2(max_edges)
        assert num_edges <= max_edges
        num_possible_edges = ((num_nodes)**2-num_nodes) * num_relations
        _bits_p_s_given_e += log2comb(num_possible_edges, num_edges)

        _compression_bits = _bits_p_e + _bits_p_s_given_e

        bits_p_e += _bits_p_e
        bits_p_s_given_e += _bits_p_s_given_e
        compression_bits += _compression_bits

    print(f"Dataset: {dataset} ({num_graphs} graphs)")
    print(f"\tAverage Compression Cost for p(S|E):", round(bits_p_s_given_e/num_graphs, 2))
    print(f"\tAverage Compression Cost for  p(E):", round(bits_p_e/num_graphs, 2))
    print(f"\tAverage Compression Cost:", round(compression_bits/num_graphs, 2))
    print(f"\n\n")

    return bits_p_s_given_e, bits_p_e, compression_bits


if __name__ == "__main__":
    compute_bits_for_synthetic_data(dataset='syn-paths')
    compute_bits_for_synthetic_data(dataset='syn-tipr')
    compute_bits_for_synthetic_data(dataset='syn-types')
    compute_bits_for_wikidata_data(dataset='wd-movies')
    compute_bits_for_wikidata_data(dataset='wd-articles')
