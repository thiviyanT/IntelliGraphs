import math
import torch
from scipy.special import gammaln
from tqdm import trange
from typing import Tuple


class UniformBaseline:
    """
    A class to calculate the compression cost in bits for a given dataset, based on the Uniform Baseline model.

    :param dataset: The name of the dataset.
    :param data: The loaded dataset.
    :param is_synthetic: A boolean indicating whether the dataset is synthetic.
    :param verbose: If True, prints additional information during processing.
    :param include_null_entity: If True, includes a null entity in the calculations.
    """
    def __init__(self, dataset: str, data, is_synthetic: bool, verbose: bool = False, include_null_entity: bool = False):
        self.dataset = dataset
        self.data = data
        self.is_synthetic = is_synthetic
        self.verbose = verbose
        self.include_null_entity = include_null_entity

    @staticmethod
    def log2comb(n: int, k: int) -> float:
        """
        Compute the logarithm base 2 of the binomial coefficient.

        .. math::
            \log_2\binom{n}{k} = \frac{\log(\Gamma(n+1)) - \log(\Gamma(n-k+1)) - \log(\Gamma(k+1))}{\log(2)}

        :param n: The total number of elements.
        :param k: The number of selected elements.
        :return: The logarithm base 2 of the binomial coefficient.
        """
        return (gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)) / math.log(2)

    def compute_bits(self) -> Tuple[float, float, float]:
        """
        Compute the total compression cost in bits for the dataset.

        Depending on whether the dataset is synthetic or not, it calculates the number of bits required to represent
        the dataset using different methods.

        :return: A tuple containing the number of bits for p(S|E), the number of bits for p(E), and the total number of bits.
        """
        if self.is_synthetic:
            return self._compute_bits_for_synthetic_data()
        else:
            return self._compute_bits_for_wikidata_data()

    def _compute_bits_for_synthetic_data(self) -> Tuple[float, float, float]:
        """
        Compute the compression cost in bits for synthetic datasets.

        The method calculates the number of bits required to represent the structure (p(S|E)) and the entities (p(E))
        of the dataset, and then sums these to get the total compression cost. We assume that the number of edges and
        nodes per subgraph is fixed, which enables us to compute the compression bits without iterating through the
        entire dataset.

        :return: A tuple containing the number of bits for p(S|E), the number of bits for p(E), and the total number of bits.
        """
        train, val, test, (e2i, i2e), (r2i, i2r), _, _ = self.data

        num_entities = len(e2i) + (1 if self.include_null_entity else 0)
        num_relations = len(i2r)
        num_edges = train.size(1)

        subjects = train[0, :, 0]
        objects = train[0, :, 2]
        concatenated = torch.cat((subjects, objects))
        unique_nodes = torch.unique(concatenated)
        num_nodes = len(unique_nodes)

        if self.verbose:
            print(f"{len(e2i)} entities")
            print(f"{len(i2r)} relations")
            print(f"{train.size(0)} training triples")
            print(f"{test.size(0)} test triples")
            print(f"{train.size(0) + test.size(0)} total triples")
            print(f"{num_edges} edges")
            print(f"{num_nodes} nodes")

        num_possible_edges = ((num_nodes) ** 2 - num_nodes) * num_relations
        bits_p_s_given_e = self.log2comb(num_possible_edges, num_edges)

        bits_p_e = self.log2comb(num_entities, num_nodes)
        compression_bits = bits_p_s_given_e + bits_p_e

        print(f"Dataset: {self.dataset}")
        print(f"\tNumber of bits for p(S|E): {round(bits_p_s_given_e, 2)}")
        print(f"\tNumber of bits for p(E): {round(bits_p_e, 2)}")
        print(f"\tTotal number of bits: {round(compression_bits, 2)}\n\n")

        return bits_p_s_given_e, bits_p_e, compression_bits

    def _compute_bits_for_wikidata_data(self) -> Tuple[float, float, float]:
        """
        Compute the compression cost in bits for Wikidata datasets.

        This method calculates the number of bits required to represent the structure (p(S|E)) and the entities (p(E))
        of each graph in the dataset, and then averages these to get the total compression cost.

        :return: A tuple containing the total number of bits for p(S|E), the total number of bits for p(E), and the total number of bits.
        """
        train, val, test, (e2i, i2e), (r2i, i2r), (min_edges, max_edges), (min_nodes, max_nodes) = self.data

        num_entities = len(e2i) + (1 if self.include_null_entity else 0)
        num_relations = len(i2r)
        num_graphs = len(test)

        if self.verbose:
            print(f"{len(e2i)} entities")
            print(f"{len(i2r)} relations")
            print(f"{test.size(0)} training triples")
            print(f"{test.size(0)} test triples")
            print(f"{test.size(0) + train.size(0)} total triples")

        bits_p_e_total, bits_p_s_given_e_total, compression_bits_total = 0, 0, 0

        for i in trange(len(test)):
            padded_graph = test[i]
            padding_triple = torch.tensor([-1, -1, -1])
            graph = padded_graph[~torch.all(padded_graph == padding_triple, dim=1)]

            subjects = graph[:, 0]
            objects = graph[:, 2]
            concatenated = torch.cat((subjects, objects))
            unique_nodes = torch.unique(concatenated)
            num_nodes = len(unique_nodes)
            num_edges = graph.size(0)

            bits_p_e = math.log2(max_nodes) + self.log2comb(num_entities, num_nodes)
            num_possible_edges = ((num_nodes) ** 2 - num_nodes) * num_relations
            bits_p_s_given_e = math.log2(max_edges) + self.log2comb(num_possible_edges, num_edges)

            compression_bits = bits_p_e + bits_p_s_given_e

            bits_p_e_total += bits_p_e
            bits_p_s_given_e_total += bits_p_s_given_e
            compression_bits_total += compression_bits

        print(f"Dataset: {self.dataset} ({num_graphs} graphs)")
        print(f"\tAverage Compression Cost for p(S|E): {round(bits_p_s_given_e_total / num_graphs, 2)}")
        print(f"\tAverage Compression Cost for p(E): {round(bits_p_e_total / num_graphs, 2)}")
        print(f"\tAverage Compression Cost: {round(compression_bits_total / num_graphs, 2)}\n\n")

        return bits_p_s_given_e_total, bits_p_e_total, compression_bits_total