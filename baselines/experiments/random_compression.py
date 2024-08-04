from _context import baselines
from util import d, tic, toc, get_slug, compute_entity_frequency, read_config
from tqdm import trange
import multiprocessing as mp
import wandb
import torch
import random
import math
from collections import defaultdict


def count_edge_frequencies(triples_tensor):
    """
    Count the frequency of edges (relations between nodes) in the given 3D tensor of triples.

    Returns:
        edge_frequencies (dict): A dictionary where the keys are (relation, head, tail) tuples and
                                 the values are the frequencies of those edges.
    """
    edge_frequencies = defaultdict(int)


    for subgraph in triples_tensor:
        for triple in subgraph:
            head, relation, tail = triple.tolist()
            edge_frequencies[(relation, head, tail)] += 1

    return edge_frequencies


# def count_edge_frequencies(triples_tensor):
#     """
#     Count the number of triples per subgraph in the given 3D tensor of triples.
#
#     Returns:
#         num_triples_per_graph (list): A list containing the number of triples for each subgraph.
#     """
#     num_triples_per_graph = []
#
#     # Iterate through each subgraph in the tensor
#     for subgraph in triples_tensor:
#         # Count the number of triples in this subgraph
#         num_triples = subgraph.size(0)
#         num_triples_per_graph.append(num_triples)
#
#     return num_triples_per_graph


def calculate_edge_probabilities(edge_frequencies, num_entities, num_relations):
    """
    Calculate the probabilities for each edge based on their frequencies in the given triples tensor.

    Returns:
        p_s_given_e (list): A list of probabilities for each edge in the form of (relation, head, tail).
    """
    # Sum of all edge frequencies
    total_edges = sum(edge_frequencies.values())

    # Initialsze probabilities for all edges
    p_s_given_e = [0] * (num_relations * num_entities * num_entities)

    # Calculate probabilities for each edge
    for (relation, head, tail), freq in edge_frequencies.items():
        # TODO: Find a more intuitive way to do this
        # Compute a unique index for the edge (relation, head, tail) in a flattened list of all possible edges.
        # The index is calculated based on the assumption that all possible (relation, head, tail) combinations
        # are stored in a flattened 1D array. This converts the 3D index of (relation, head, tail)
        # into a single unique index in a 1D array.
        index = relation * num_entities * num_entities + head * num_entities + tail
        p_s_given_e[index] = freq / total_edges

    return p_s_given_e


def compute_bits(probabilities):
    """ Compute the exact number of bits required to represent a list of probabilities """
    total_bits = 0
    for prob in probabilities:
        print(prob)
        num_bits = -math.log2(prob)
        total_bits += num_bits
    return total_bits


def random_graph_prediction(num_entities, num_relations, number_nodes, edge_frequencies):
    """
    Generate a random graph prediction and compute the number of bits required
    """

    # Generate probabilities for p(E) uniformly
    # p_e = [random.random() for _ in range(num_entities)]
    p_e = [(1 / num_entities) for _ in range(number_nodes)]

    # Calculate probabilities for p(S|E) using edge frequencies
    # p_s_given_e = [random.random() for _ in range(num_relations * number_nodes * number_nodes)]
    p_s_given_e = [0.5 for _ in range(num_relations * number_nodes * number_nodes)]  # TODO: Count the frequency of edges in the training data
    # p_s_given_e = calculate_edge_probabilities(edge_frequencies, num_entities, num_relations)

    # Compute the number of bits required to store graphs
    bits_p_e = compute_bits(p_e)
    bits_p_s_given_e = compute_bits(p_s_given_e)

    return bits_p_e, bits_p_s_given_e, (bits_p_e + bits_p_s_given_e)


def train(wandb):
    """ Train baseline models on a dataset """

    config = wandb.config

    train, val, test, (n2i, i2n), (r2i, i2r) = \
        baselines.load(config["dataset"], padding=True)

    if config["final"]:
        train, test = torch.cat([train, val], dim=0), test
    else:
        train, test = train, val

    edge_frequencies = count_edge_frequencies(train)

    print(len(i2n), 'nodes')
    print(len(i2r), 'relations')
    print(train.size(0), 'training triples')
    print(test.size(0), 'test triples')
    print(train.size(0) + test.size(0), 'total triples')

    for e in range(1):
        _bits_p_e, _bits_p_s_given_e, _compression_bits = 0, 0, 0

        for fr in trange(0, train.size(0), config["batch-size"]):
            to = min(train.size(0), fr + config["batch-size"])
            positives = train[fr:to].to(d())

            assert len(positives.size()) == 3
            bs, _, _ = positives.size()

            for b, subgraph in enumerate(positives):

                if config["padding"]:
                    subgraph = subgraph[subgraph[:, 1] != -1]  # Remove padding

                # Map global indices to local indices for entities
                entities = torch.unique(torch.cat([subgraph[:, 0], subgraph[:, 2]])).tolist()

                num_entities = len(entities)
                num_relations = len(i2r)

                bits_p_e, bits_p_s_given_e, compression_bits = random_graph_prediction(len(i2n), num_relations, num_entities, edge_frequencies)

                _bits_p_e += bits_p_e
                _bits_p_s_given_e += bits_p_s_given_e
                _compression_bits += compression_bits

        print("Number of bits for p(S|E):", round(_bits_p_s_given_e / len(test), 2))  # This is a mistake
        print("Number of bits for p(E):", round(_bits_p_e/len(test), 2))  # This is a mistake
        print("Total number of bits:", round(_compression_bits/len(test), 2))  # This is a mistake

        break


if __name__ == "__main__":

    mp.set_start_method('spawn')

    # Default configuration
    hyperparameter_defaults = {
        "dataset": 'syn-paths',  # Name of the dataset
        "final": True,  # Whether to use the final test set
        "batch-size": 1024,  # Batch size
        "padding": False,
    }

    wandb.init(
        project="kgi",
        entity="nesy-gems",
        name=f"Sampling Test",
        notes="",
        tags=['sampling'],
        config=hyperparameter_defaults,
    )

    print('Hyper-parameters: ', wandb.config)
    train(wandb)
