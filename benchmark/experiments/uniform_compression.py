import math
import torch
from scipy.special import gammaln
from tqdm import trange
from _context import models
from typing import Tuple


def log2comb(n: int, k: int) -> float:
    """
    Compute the logarithm base 2 of the binomial coefficient.

    This computes the number of ways to choose `k` edges from `n` edges without replacement,
    avoiding overflow by using the gammaln function.

    This is functionally equivalent to doing the following operation:
        bits_p_s_given_e = math.log2(math.comb(n, k))

    :param n: Total number of items.
    :param k: Number of selected items.
    :return: log2 of the binomial coefficient.
    """
    return (gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)) / math.log(2)


def compute_bits_for_synthetic_data(dataset: str, verbose: bool = False, include_null_entity: bool = False) -> Tuple[
    float, float, float]:
    """
    Estimate the compression bits for storing graphs sampled using a uniform distribution.

    :param dataset: The dataset to evaluate.
    :param verbose: If True, print detailed statistics.
    :param include_null_entity: If True, include a null entity in the count.
    :return: A tuple containing the bits for p(S|E), p(E), and total compression bits.
    """
    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = models.load_data(dataset, padding=True)

    num_entities = len(e2i) + (1 if include_null_entity else 0)
    num_relations = len(i2r)
    num_edges = train.size(1)

    # Here we assume that all the triples are similar to the first triple
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
    num_possible_edges = ((num_nodes) ** 2 - num_nodes) * num_relations
    bits_p_s_given_e = log2comb(num_possible_edges, num_edges)

    bits_p_e = log2comb(num_entities, num_nodes)
    compression_bits = bits_p_s_given_e + bits_p_e

    print(f"Dataset: {dataset}")
    print(f"\tNumber of bits for p(S|E):", round(bits_p_s_given_e, 2))
    print(f"\tNumber of bits for p(E):", round(bits_p_e, 2))
    print(f"\tTotal number of bits:", round(compression_bits, 2))
    print(f"\n\n")

    return bits_p_s_given_e, bits_p_e, compression_bits


def compute_bits_for_wikidata_data(dataset: str, verbose: bool = False, include_null_entity: bool = False) -> Tuple[
    float, float, float]:
    """
    Estimate the compression bits for storing graphs sampled using a uniform distribution.

    :param dataset: The dataset to evaluate.
    :param verbose: If True, print detailed statistics.
    :param include_null_entity: If True, include a null entity in the count.
    :return: A tuple containing the bits for p(S|E), p(E), and total compression bits.
    """
    train, val, test, (e2i, i2e), (r2i, i2r), (min_edges, max_edges), (min_nodes, max_nodes) = models.load_data(dataset,
                                                                                                                padding=True)

    num_entities = len(e2i) + (1 if include_null_entity else 0)  # Add one for null entities if specified
    num_relations = len(i2r)
    num_graphs = len(test)

    if verbose:
        print(len(e2i), 'entities')
        print(len(i2r), 'relations')
        print(test.size(0), 'training triples')
        print(test.size(0), 'test triples')
        print(test.size(0) + train.size(0), 'total triples')

    bits_p_e_total, bits_p_s_given_e_total, compression_bits_total = 0, 0, 0

    # For every graph, compute the compression bits - number of nodes and edges would vary between graphs
    for i in trange(len(test)):
        padded_graph = test[i]

        # Filter out rows that are not equal to the padding triple
        padding_triple = torch.tensor([-1, -1, -1])
        graph = padded_graph[~torch.all(padded_graph == padding_triple, dim=1)]

        subjects = graph[:, 0]
        objects = graph[:, 2]
        concatenated = torch.cat((subjects, objects))
        unique_nodes = torch.unique(concatenated)
        num_nodes = len(unique_nodes)
        num_edges = graph.size(0)

        # Explain in the paper that for simplicity we take the maximum nodes from the data (train, val, test)
        bits_p_e = math.log2(max_nodes) + log2comb(num_entities, num_nodes)

        # Calculate bits for p(S|E)
        num_possible_edges = ((num_nodes) ** 2 - num_nodes) * num_relations
        bits_p_s_given_e = math.log2(max_edges)
        assert num_edges <= max_edges
        bits_p_s_given_e += log2comb(num_possible_edges, num_edges)

        compression_bits = bits_p_e + bits_p_s_given_e

        bits_p_e_total += bits_p_e
        bits_p_s_given_e_total += bits_p_s_given_e
        compression_bits_total += compression_bits

    print(f"Dataset: {dataset} ({num_graphs} graphs)")
    print(f"\tAverage Compression Cost for p(S|E):", round(bits_p_s_given_e_total / num_graphs, 2))
    print(f"\tAverage Compression Cost for  p(E):", round(bits_p_e_total / num_graphs, 2))
    print(f"\tAverage Compression Cost:", round(compression_bits_total / num_graphs, 2))
    print(f"\n\n")

    return bits_p_s_given_e_total, bits_p_e_total, compression_bits_total


if __name__ == "__main__":
    compute_bits_for_synthetic_data(dataset='syn-paths')
    compute_bits_for_synthetic_data(dataset='syn-tipr')
    compute_bits_for_synthetic_data(dataset='syn-types')
    compute_bits_for_wikidata_data(dataset='wd-movies')
    compute_bits_for_wikidata_data(dataset='wd-articles')
