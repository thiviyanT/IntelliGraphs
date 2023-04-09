import random
from typing import List, Tuple, Dict
import os
import zipfile
from tqdm import tqdm
from graphviz import Source


class IntelliGraphs:
    def __init__(
            self,
            random_seed: int = 42,
            num_graphs: int = 1,
            num_triples: int = 100,
            var_length: bool = False,
            min_triples: int = 3,
            max_triples: int = 20,
            split_ratio: Tuple[float, float, float] = (0.6, 0.3, 0.1)
    ):
        """
        Initialize the IntelliGraphs instance.

        Args:
            random_seed (int): The seed for the random generation. Default is 42.
            num_graphs (int): The number of graphs to generate. Default is 1.
            num_triples (int): The number of random triples to generate per graph. Default is 100.
            var_length (bool): If True, ignore num_triples and generates graphs with variable length
                               of triples between 3 and 20. Default is False.
            min_triples (int): The minimum number of triples to generate if var_length is True. Default is 3.
            max_triples (int): The maximum number of triples to generate if var_length is True. Default is 20.
            split_ratio (Tuple[float, float, float]): The ratio of the train, valid, and test sets. Default is (0.6, 0.3, 0.1).
        """
        self.random_seed = random_seed
        self.num_graphs = num_graphs
        self.num_triples = num_triples
        self.var_length = var_length
        self.min_triples = min_triples
        self.max_triples = max_triples
        self.split_ratio = split_ratio
        random.seed(self.random_seed)

        self.graphs = None
        self.splits = None

    def get_graphs(self):
        """
        Get the list of graphs.

        Returns:
            List[List[Tuple[str, str, str]]]: A list of lists of triples, one list for each graph.
        """
        assert self.graphs is not None, "Graphs have not been generated. Call generate_graphs() first."
        return self.graphs

    def get_splits(self):
        """
        Get the data splits.

        Returns:
            Dict[str, List[List[Tuple[str, str, str]]]]: A dictionary with the train, valid, and test splits.
        """
        assert self.splits is not None, "Data splits have not been generated. Call split_data() first."
        return self.splits

    def generate_graphs(self):
        """
        Generate random triples for the specified number of graphs.
        """
        self.graphs = [self.generate_random_triples() for _ in tqdm(range(self.num_graphs), desc="Generating graphs")]

    def generate_random_triples(self) -> List[Tuple[str, str, str]]:
        """
        Generate random triples to create a synthetic Knowledge Graph.

        Returns:
            List[Tuple[str, str, str]]: A list of random triples.
        """
        subjects = ["Alice", "Bob", "Charlie", "David", "Eve"]
        predicates = ["likes", "dislikes", "knows", "loves", "hates"]
        objects = ["pizza", "ice cream", "sushi", "football", "music"]

        triples = []
        num_triples = self.num_triples

        if self.var_length:
            num_triples = random.randint(self.min_triples, self.max_triples)

        for _ in range(num_triples):
            subject = random.choice(subjects)
            predicate = random.choice(predicates)
            obj = random.choice(objects)
            triples.append((subject, predicate, obj))

        return triples

    def to_natural_language(self) -> List[List[str]]:
        """
        Generate a list of natural language sentences representing the triples.

        Returns:
            List[List[str]]: A list of lists of natural language sentences, one list for each graph.
        """
        assert self.graphs is not None, "Graphs have not been generated. Call generate_graphs() first."
        all_sentences = []

        for graph in tqdm(self.graphs, desc="Converting to natural language"):
            sentences = []
            for triple in graph:
                subject, predicate, obj = triple
                sentence = f"{subject} {predicate} {obj}."
                sentences.append(sentence)
            all_sentences.append(sentences)

        return all_sentences

    @staticmethod
    def check_transductive_features(data: Dict[str, List[List[Tuple[str, str, str]]]]) -> Dict[
        str, List[List[Tuple[str, str, str]]]]:
        """
        THIS COULD BE AN EXPENSIVE OPERATION IF THE NUMBER OF TRIPLES OR/AND THE NUMBER OF GRAPHS IS LARGE.

        Check if entities and relations appear in train/valid and test sets.
        If they don't, remove entire graphs from the train/valid/test sets.
        Raise an error if the number of graphs in the train/valid/test sets is less than 1.

        Args:
            data (Dict[str, List[List[Tuple[str, str, str]]]]): A dictionary containing the train, valid, and test sets.

        Returns:
            Dict[str, List[List[Tuple[str, str, str]]]]: The updated data dictionary with the removed graphs.
        """
        unique_entities = set()
        unique_relations = set()

        # Collect unique entities and relations from the training set
        for graph in data['train']:
            for triple in graph:
                unique_entities.add(triple[0])
                unique_entities.add(triple[2])
                unique_relations.add(triple[1])

        for split_name in ['valid', 'test']:
            removed_graphs = 0
            new_graphs = []
            for graph in data[split_name]:
                if all(triple[0] in unique_entities and triple[2] in unique_entities and triple[1] in unique_relations
                       for triple in graph):
                    new_graphs.append(graph)
                else:
                    removed_graphs += 1

            if len(new_graphs) < 1:
                raise ValueError(f"Error: The number of graphs in the {split_name} set is less than 1.")

            data[split_name] = new_graphs
            print(f"Removed {removed_graphs} graphs from the {split_name} set due to missing entities/relations.")

        return data

    def split_data(self, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   check_transductive_overlap=False) -> Dict[str, List[List[Tuple[str, str, str]]]]:
        """
        Split the knowledge graphs into train, valid, and test sets according to the split ratio.

        Args:
            split_ratio (Tuple[float, float, float]): A tuple with 3 float values representing the ratio for
                                                      train, valid, and test splits, respectively.
                                                      Default is (0.8, 0.1, 0.1).
            check_transductive_overlap (bool): If True, check if entities and relations appear in train/valid
                                               and test sets. If they don't, remove those triples from the valid
                                               and test sets.

        Returns:
            Dict[str, List[List[Tuple[str, str, str]]]]: A dictionary with keys 'train', 'valid', and 'test'
                                                         containing the respective knowledge graph splits.
        """
        assert self.graphs is not None, "Graphs have not been generated. Call generate_graphs() first."

        # Remove duplicate graphs
        unique_graphs = list(set(map(tuple, self.graphs)))
        removed_graphs = len(self.graphs) - len(unique_graphs)
        if removed_graphs > 0:
            print(f"{removed_graphs} duplicate graphs removed.")

        train_ratio, valid_ratio, test_ratio = split_ratio
        total_graphs = len(self.graphs)

        train_size = int(total_graphs * train_ratio)
        valid_size = int(total_graphs * valid_ratio)

        train_graphs = self.graphs[:train_size]
        valid_graphs = self.graphs[train_size: train_size + valid_size]
        test_graphs = self.graphs[train_size + valid_size:]

        data = {"train": train_graphs, "valid": valid_graphs, "test": test_graphs}

        # Check entities and relations are present in the training set, valid and test sets.
        # If not, remove those graphs from the valid and test sets.
        if check_transductive_overlap:
            data = self.check_transductive_features(data)

        # Update the splits class attribute
        self.splits = data
        return self.splits

    @staticmethod
    def _save_data(data, filename: str, file_path: str = 'output', zip_compression: bool = False) -> None:
        os.makedirs(file_path, exist_ok=True)

        def format_triple(triple: Tuple[str, str, str]) -> str:
            return f"{triple[0]} {triple[1]} {triple[2]}"

        if zip_compression:
            zip_file_name = os.path.join(file_path, f"{filename}.zip")
            with zipfile.ZipFile(zip_file_name, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                with zf.open(f"{filename}.txt", mode="w") as f:
                    for graph in data:
                        formatted_triples = [format_triple(triple) for triple in graph]
                        f.write(("\n".join(formatted_triples) + "\n\n").encode())
        else:
            file_name = os.path.join(file_path, f"{filename}.txt")
            with open(file_name, "w") as f:
                for graph in data:
                    formatted_triples = [format_triple(triple) for triple in graph]
                    f.write("\n".join(formatted_triples) + "\n\n")

    def save_graphs(self, filename: str = 'knowledge_graphs', file_path: str = 'output',
                    zip_compression: bool = False) -> None:
        """
        Save the knowledge graphs to a single text file under the specified directory.

        Args:
            filename (str): The name of the file to save the knowledge graphs. Default is 'knowledge_graphs'.
            file_path (str): The directory under which to store the file. Default is 'output'.
            zip_compression (bool): If True, the output is zip compressed. Default is False.
        """
        assert self.graphs is not None, "Graphs have not been generated. Call generate_graphs() first."
        self._save_data(self.graphs, filename, file_path, zip_compression)

    def save_splits(self, filename: str = 'knowledge_graphs', file_path: str = 'output',
                    zip_compression: bool = False) -> None:
        """
        Save the train, valid, and test splits to separate text files under the specified directory.

        Args:
            file_path (str): The directory under which to store the files. Default is 'output'.
            zip_compression (bool): If True, the output is zip compressed. Default is False.
        """
        assert self.splits is not None, "Data splits have not been generated. Call split_data() first."

        for split_name, data in self.splits.items():
            self._save_data(data, f"{filename}_{split_name}", file_path, zip_compression)

    @staticmethod
    def print_graph(graph: List[Tuple[str, str, str]]) -> str:
        """
        Print and return the given graph in DOT format.

        Args:
            graph (List[Tuple[str, str, str]]): The graph to be printed.

        Returns:
            str: The graph in DOT format.
        """
        dot_format = "digraph {\n"

        for triple in graph:
            subject, predicate, obj = triple
            dot_format += f'    "{subject}" -> "{obj}" [label="{predicate}"];\n'

        dot_format += "}\n"

        print(dot_format)
        return dot_format

    def visualize_graph(self, graph: List[Tuple[str, str, str]]) -> None:
        """
        Visualize the given graph using the graphviz package.

        Args:
            graph (List[Tuple[str, str, str]]): The graph to be visualized.
        """
        dot_format = self.print_graph(graph)
        src = Source(dot_format)
        src.render(format='png', filename='graph', cleanup=True, view=True)
