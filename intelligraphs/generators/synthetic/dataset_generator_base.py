import os
import zipfile
from typing import Dict, List, Tuple
from graphviz import Source
import random
from tqdm import tqdm
import argparse

class BaseSyntheticDatasetGenerator:
    def __init__(self, train_size, val_size, test_size, num_edges, random_seed, dataset_name):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_edges = num_edges
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        random.seed(self.random_seed)

    def sample_synthetic_data(self, num_graphs):
        """Sample synthetic data - This method should be implemented by the child classes."""
        raise NotImplementedError("This method should be implemented by the child class.")

    def split_dataset(self, triples):
        """Split dataset into training, validation, and test sets."""
        train_offset = self.train_size
        val_offset = self.train_size + self.val_size

        return {
            'train': triples[:train_offset],
            'valid': triples[train_offset:val_offset],
            'test': triples[val_offset:val_offset + self.test_size],
        }

    def check_unique_graphs(self, data: Dict[str, List[List[Tuple[str, str, str]]]]) -> None:
        """Check if all graphs in train, validation, and test splits are unique to prevent data leakage."""
        all_graphs = set()

        for split_name in ['train', 'valid', 'test']:
            for graph in data[split_name]:
                graph_tuple = tuple(sorted(graph))  # Sort the graph to avoid order issues
                if graph_tuple in all_graphs:
                    raise ValueError(f"Duplicate graph found in {split_name} set. Data leakage detected.")
                all_graphs.add(graph_tuple)

        print("All graphs in train, validation, and test splits are unique.")

    def save_to_tsv(self, triples, filename):
        """Save triples to a TSV file with a space between each graph."""
        with open(filename, 'w') as f:
            for graph in triples:
                for triple in graph:
                    s, p, o = triple
                    f.write(f"{s}\t{p}\t{o}\n")
                f.write("\n")  # Space between each graph

    def zip_output_files(self):
        """Compress the output TSV files into a ZIP file."""
        zip_filename = f"{self.dataset_name}.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in ["train_split.tsv", "val_split.tsv", "test_split.tsv"]:
                if os.path.exists(file):
                    zipf.write(file)
                    os.remove(file)  # Optionally remove the file after zipping
        print(f"Compressed output files into {zip_filename}")

    def generate_and_save(self):
        """Generate synthetic data, split it, check for uniqueness, and save it."""
        num_graphs = self.train_size + self.val_size + self.test_size
        print("Generating data...")
        triples = self.sample_synthetic_data(num_graphs=num_graphs)

        print("Splitting dataset...")
        split = self.split_dataset(triples)

        # Check for unique graphs
        self.check_unique_graphs(split)

        # Check transductive features
        split = self.check_transductive_features(split)

        print("Saving data to files...")
        self.save_to_tsv(split['train'], "train_split.tsv")
        self.save_to_tsv(split['valid'], "val_split.tsv")
        self.save_to_tsv(split['test'], "test_split.tsv")

        # Zip the output files
        self.zip_output_files()
        print('Done')

    @staticmethod
    def check_transductive_features(data: Dict[str, List[List[Tuple[str, str, str]]]]) -> Dict[
        str, List[List[Tuple[str, str, str]]]]:
        """
        Check if entities and relations appear in train/valid and test sets.
        If they don't, remove entire graphs from the train/valid/test sets.
        Raise an error if the number of graphs in the train/valid/test sets is less than 1.
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

    @staticmethod
    def print_graph(graph: List[Tuple[str, str, str]]) -> str:
        """
        Print and return the given graph in DOT format.
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
        """
        dot_format = self.print_graph(graph)
        src = Source(dot_format)
        src.render(format='png', filename='graph', cleanup=True, view=True)
