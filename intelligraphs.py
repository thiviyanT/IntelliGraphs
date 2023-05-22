import random
from typing import List, Tuple, Dict
from tqdm import tqdm
from graphviz import Source
from pyparsing import Word, alphas, Group, Optional, Suppress, delimitedList
import requests
from pathlib import Path
import urllib.request
import os
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader


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


class IntelliGraphsDataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.base_dir = '.data'

        # TODO: Replace with Zenodo links
        self.download_links = {
            'syn-paths': 'https://www.dropbox.com/s/0mlyw0nkjbbucl5/syn-paths.zip?dl=1',
            'syn-tipr': 'https://www.dropbox.com/s/sjx8cvj49w24qk0/syn-tipr.zip?dl=1',
            'syn-types': 'https://www.dropbox.com/s/7h44h9g138ylk5u/syn-types.zip?dl=1',
            'wd-articles': 'https://www.dropbox.com/s/iftjkvivu6owxhy/wd-articles.zip?dl=1',
            'wd-movies': 'https://www.dropbox.com/s/nf7vtco1t4jn0o7/wd-movies.zip?dl=1'
        }

        # Download dataset if not exists
        self.download_dataset()

    def create_data_folder(self):
        """Create data folder if not exists."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"Created '{self.base_dir}' folder.")

    def download_dataset(self):
        """Download dataset if not exists."""
        dataset_folder = self.dataset_name
        dataset_folder_path = os.path.join(self.base_dir, dataset_folder)
        dataset_zip = self.dataset_name + '.zip'
        dataset_zip_path = os.path.join(self.base_dir, dataset_zip)

        self.create_data_folder()

        if not os.path.exists(dataset_folder_path):
            print(f'Dataset {self.dataset_name} not found. Downloading...')

            download_link = self.download_links.get(self.dataset_name)
            if download_link is None:
                print(f'Dataset {self.dataset_name} is not available.')
                return

            os.makedirs(self.base_dir, exist_ok=True)
            urllib.request.urlretrieve(download_link, dataset_zip_path)
            print(f'Dataset {self.dataset_name} downloaded successfully.')

            self.extract_dataset(dataset_zip_path)

            os.remove(dataset_zip_path)
            print(f"Removed '{dataset_zip_path}' file.")

    def extract_dataset(self, dataset_path):
        print(f'Extracting {self.dataset_name} dataset...')

        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(self.base_dir)

        print(f'{self.dataset_name} dataset extracted successfully.')


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

    def parse_fol_rules(self, file_path: str) -> List[str]:
        """
        Read FOL rules from a text file.

        Args:
            file_path (str): The path to the file containing the FOL rules.

        Returns:
            List[str]: A list of FOL rules.
        """
        # Define the grammar for parsing first-order logic rules
        identifier = Word(alphas)
        predicate = identifier
        argument = identifier

        predicate_expression = Group(predicate + Suppress("(") + delimitedList(argument) + Suppress(")"))
        quantifier = Group(Word("∀∃")("quantifier") + identifier("variable"))
        implication = Suppress("→")
        expression = Group(
            Optional(quantifier) + predicate_expression("left") + Optional(implication + predicate_expression("right")))

        def read_rules_from_file(filename):
            with open(filename, "r") as file:
                content = file.readlines()
            return [rule.strip() for rule in content]

        filename = file_path
        rules = read_rules_from_file(filename)
        return rules

    @staticmethod
    def download_dataset(dataset_name: str, output_filepath: str) -> None:
        """
        Download a dataset from Zenodo.

        Args:
            dataset_name (str): The name of the dataset to be downloaded.
            output_filepath (str): The output filepath to save the downloaded dataset.
        """
        # Define the Zenodo base URL and construct the dataset URL
        zenodo_base_url = "https://zenodo.org/record/"
        dataset_url = f"{zenodo_base_url}{dataset_name}/files/{dataset_name}.zip"

        # Download the dataset
        response = requests.get(dataset_url)
        response.raise_for_status()

        # Save the dataset to the output filepath
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as output_file:
            output_file.write(response.content)

        print(f"Dataset '{dataset_name}' downloaded successfully to {output_filepath}")


class CustomDataset(Dataset):
    """ Custom dataset class. """

    def __init__(self, file_path, node_mapping, relation_mapping, padding=True, max_graph_size=None):
        self.data = []

        self.data = load_strings(file_path, split_tab=True)
        self.data = split_subgaphs(self.data)
        self.data = map_nodes_relations(self.data, node_mapping, relation_mapping)
        if padding:
            if max_graph_size is None:
                raise ValueError('max_graph_size must be specified if padding is True.')
            self.data = pad_data(self.data, max_graph_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class IntelliGraphsDataLoader:
    """ DataLoader for IntelliGraphs datasets. """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.base_dir = '.data'

        # TODO: Replace with Zenodo links
        self.download_links = {
            'syn-paths': 'https://www.dropbox.com/s/kp1xp2rbieib4gl/syn-paths.zip?dl=1',
            'syn-tipr': 'https://www.dropbox.com/s/wgm2yr7h8dhcj52/syn-tipr.zip?dl=1',
            'syn-types': 'https://www.dropbox.com/s/yx7vrvsxme53xce/syn-types.zip?dl=1',
            'wd-articles': 'https://www.dropbox.com/s/37etzy2pkix84o8/wd-articles.zip?dl=1',
            'wd-movies': 'https://www.dropbox.com/s/gavyilqy1kb750f/wd-movies.zip?dl=1'
        }

        # Download dataset if not exists
        self.download_dataset()

    def create_data_folder(self):
        """
        Create directory for storing data, if not exists.

        Returns:
            None

        """
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"Created '{self.base_dir}' folder.")

    def download_dataset(self):
        """
        Download dataset if they do not already exist locally.

        Returns:
            None
        """
        dataset_folder = self.dataset_name
        dataset_folder_path = os.path.join(self.base_dir, dataset_folder)
        dataset_zip = self.dataset_name + '.zip'
        dataset_zip_path = os.path.join(self.base_dir, dataset_zip)

        self.create_data_folder()

        if not os.path.exists(dataset_folder_path):
            print(f'Dataset {self.dataset_name} not found. Downloading...')

            download_link = self.download_links.get(self.dataset_name)
            if download_link is None:
                print(f'Dataset {self.dataset_name} is not available.')
                return

            os.makedirs(self.base_dir, exist_ok=True)
            urllib.request.urlretrieve(download_link, dataset_zip_path)
            print(f'Dataset {self.dataset_name} downloaded successfully.')

            self.extract_dataset(dataset_zip_path)

            os.remove(dataset_zip_path)
            print(f"Removed '{dataset_zip_path}' file.")

    def extract_dataset(self, dataset_path):
        """
        Unzip dataset.

        Args:
            dataset_path (str): Path to dataset zip file.

        Returns:
            None
        """
        print(f'Extracting {self.dataset_name} dataset...')

        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(self.base_dir)

        print(f'{self.dataset_name} dataset extracted successfully.')

    def load_torch(self, batch_size=32, padding=True, shuffle_train=False, shuffle_valid=False, shuffle_test=False,):
        """
        Load dataset as torch tensors for PyTorch.

        Args:
            batch_size (int): Batch size.
            padding (bool): Pad subgraphs with empty triples [-1, -1, -1].
            shuffle_train (bool): Shuffle training data.
            shuffle_valid (bool): Shuffle validation data.
            shuffle_test (bool): Shuffle test data.

        Returns:
            (train_loader, valid_loader, test_loader): PyTorch data loaders.
        """
        dataset_folder = self.dataset_name
        dataset_folder_path = os.path.join(self.base_dir, dataset_folder)

        train_file = os.path.join(dataset_folder_path, 'train.txt')
        valid_file = os.path.join(dataset_folder_path, 'valid.txt')
        test_file = os.path.join(dataset_folder_path, 'test.txt')

        (n2i, i2n), (r2i, i2r), max_len = create_mapping(train_file, valid_file, test_file)

        # Create custom datasets
        train_dataset = CustomDataset(train_file, n2i, r2i, padding=padding, max_graph_size=max_len)
        valid_dataset = CustomDataset(valid_file, n2i, r2i, padding=padding, max_graph_size=max_len)
        test_dataset = CustomDataset(test_file, n2i, r2i, padding=padding, max_graph_size=max_len)

        train_dataset = torch.tensor(train_dataset)
        valid_dataset = torch.tensor(valid_dataset)
        test_dataset = torch.tensor(test_dataset)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_valid)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

        return train_loader, valid_loader, test_loader
