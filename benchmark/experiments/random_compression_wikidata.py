from _context import models
from tqdm import trange
import torch
import math


def compute_bits(dataset):
    """ Estimate the compression bits for storing graphs sampled using uniform distribution """

    train, val, test, (e2i, i2e), (r2i, i2r), (min_edges, max_edges), (min_nodes, max_nodes) = (
        models.load_data(dataset, padding=True))

    # TODO: Peter, do we iterate through all the splits or just train and validation?
    data = torch.cat([train, val], dim=0)

    # TODO: Peter, the concept of null entity is only True for the VAE. The KGE models do not use it
    num_entities = len(e2i) + 1  # Add one for null entities
    num_relations = len(i2r)
    num_graphs = len(data)

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

        _bits_p_e += math.log2(max_nodes)
        _bits_p_e += math.log2(num_entities) * num_nodes

        _bits_p_s_given_e += math.log2(max_edges)
        assert num_edges <= max_edges

        _compression_bits = _bits_p_e + _bits_p_s_given_e

        bits_p_e += _bits_p_e
        bits_p_s_given_e += _bits_p_s_given_e
        compression_bits += _compression_bits

    print(f"Dataset: {dataset} ({num_graphs} graphs)")
    print(f"\tAverage Compression Cost for p(S|E):", round(bits_p_s_given_e/num_graphs, 2))
    print(f"\tAverage Compression Cost for  p(E):", round(bits_p_e/num_graphs, 2))
    print(f"\tAverage Compression Cost:", round(compression_bits/num_graphs, 2))
    print(f"\n\n")


if __name__ == "__main__":
    compute_bits(dataset='wd-movies')
    compute_bits(dataset='wd-articles')
