from _context import baselines
import torch
import math


def compute_bits(dataset):
    """ Estimate the compression bits for storing graphs sampled using uniform distribution """

    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = baselines.load_data(dataset, padding=True)

    # TODO: Peter, the concept of null entity is only True for the VAE. The KGE baselines do not use it
    num_entities = len(e2i) + 1  # Add one for null entities
    num_relations = len(i2r)
    num_edges = train.size(1)

    # Here is we assume the all the triples are similar to the first triple
    subjects = train[0, :, 0]
    objects = train[0, :, 2]
    concatenated = torch.cat((subjects, objects))
    unique_nodes = torch.unique(concatenated)
    num_nodes = len(unique_nodes)

    print(len(e2i), 'entities')
    print(len(i2r), 'relations')
    print(train.size(0), 'training triples')
    print(test.size(0), 'test triples')
    print(train.size(0) + test.size(0), 'total triples')
    print(num_edges, 'edges')
    print(num_nodes, 'nodes')

    # Model assumes that self-loops are not allowed and edge independence
    num_possible_edges = ((num_nodes)**2-num_nodes) * num_relations
    bits_p_s_given_e = math.log2(num_possible_edges) * num_relations
    bits_p_e = math.log2(num_entities) * num_nodes

    print(f"Dataset: {dataset}")
    print(f"\tNumber of bits for p(S|E):", round(bits_p_s_given_e, 2))
    print(f"\tNumber of bits for p(E):", round(bits_p_e, 2))
    print(f"\tTotal number of bits:", round(bits_p_e + bits_p_s_given_e, 2))
    print(f"\n\n")


if __name__ == "__main__":
    compute_bits(dataset='syn-paths')
    compute_bits(dataset='syn-tipr')
    compute_bits(dataset='syn-types')
