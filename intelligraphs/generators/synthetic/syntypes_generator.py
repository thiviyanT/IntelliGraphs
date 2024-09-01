from intelligraphs.generators.synthetic.dataset_generator_base import BaseSyntheticDatasetGenerator, tqdm, random, argparse
from intelligraphs.domains.SynTypes.entities import countries, cities, languages
from intelligraphs.domains.SynTypes import relations
from collections import defaultdict, deque

class SynTypesGenerator(BaseSyntheticDatasetGenerator):
    def __init__(self, train_size=60000, val_size=20000, test_size=20000, num_edges=3, random_seed=42, dataset_name="syn-types"):
        super().__init__(train_size, val_size, test_size, num_edges, random_seed, dataset_name)
        # Specific data for this generator
        self.countries = list(countries)
        self.languages = list(languages)
        self.cities = list(cities)
        self.relations = list(relations)
        self.entities = self.countries + self.languages + self.cities

    def sample_synthetic_data(self, num_graphs=5):
        """Generate synthetic data specific to this generator."""
        random.shuffle(self.entities)
        triples = set()

        with tqdm(total=num_graphs) as pbar:
            while len(triples) < num_graphs:
                subgraph = []

                while len(subgraph) < self.num_edges:
                    # Generate the first city-country pair
                    city1 = random.choice(self.cities)
                    country1 = random.choice(self.countries)
                    if city1 != country1:  # Avoid self-loops
                        subgraph.append((city1, 'could_be_part_of', country1))

                    # Generate the language spoken in the first country
                    language1 = random.choice(self.languages)
                    if language1 != country1:  # Avoid self-loops
                        subgraph.append((language1, 'spoken_in', country1))

                    # Generate a synonym for the first country
                    country1_synonym = random.choice(self.countries)
                    if country1 != country1_synonym:  # Avoid self-loops
                        subgraph.append((country1, 'same_type_as', country1_synonym))

                    # Generate the second city-country pair in the same country
                    city2 = random.choice(self.cities)
                    if city2 != country1:  # Avoid self-loops
                        subgraph.append((city2, 'could_be_part_of', country1))

                    # Generate the language spoken in the second city
                    language2 = random.choice(self.languages)
                    if language2 != country1:  # Avoid self-loops
                        subgraph.append((language2, 'spoken_in', country1))

                    # Step 6: Generate a synonym for the second city
                    city2_synonym = random.choice(self.cities)
                    if city2 != city2_synonym:  # Avoid self-loops
                        subgraph.append((city2, 'same_type_as', city2_synonym))

                    # Generate the language spoken in the second country
                    country2 = random.choice(self.countries)
                    language3 = random.choice(self.languages)
                    if language3 != country2:  # Avoid self-loops
                        subgraph.append((language3, 'spoken_in', country2))

                    # Generate a synonym for the second country
                    country2_synonym = random.choice(self.countries)
                    if country2 != country2_synonym:  # Avoid self-loops
                        subgraph.append((country2, 'same_type_as', country2_synonym))

                    # Generate the third city-country pair in the second country
                    city3 = random.choice(self.cities)
                    if city3 != country2:  # Avoid self-loops
                        subgraph.append((city3, 'could_be_part_of', country2))

                    # Generate a synonym for the third city
                    city3_synonym = random.choice(self.cities)
                    if city3 != city3_synonym:  # Avoid self-loops
                        subgraph.append((city3, 'same_type_as', city3_synonym))

                    # Check if the graph is connected
                    if self.check_location_graph(subgraph) and self.is_connected(subgraph):
                        subgraph = tuple(subgraph)
                        triples.add(subgraph)
                        pbar.update(1)

            return list(triples)

    def is_valid_triple(self, s, p, o):
        """Check if a triple is semantically valid."""
        if s in self.languages and p == 'spoken_in' and o in self.countries:
            return True
        elif s in self.cities and p == 'could_be_part_of' and o in self.countries:
            return True
        elif s in self.languages and p == 'same_type_as' and o in self.languages:
            return True
        elif s in self.cities and p == 'same_type_as' and o in self.cities:
            return True
        elif s in self.countries and p == 'same_type_as' and o in self.countries:
            return True
        else:
            return False

    def is_connected(self, graph):
        """Check if the graph is connected."""
        adj_list = defaultdict(set)

        # Build the adjacency list
        for s, p, o in graph:
            adj_list[s].add(o)
            adj_list[o].add(s)

        # Perform BFS/DFS to check connectivity
        visited = set()
        queue = deque([next(iter(adj_list))])  # Start from any node

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(adj_list[node] - visited)

        # If the number of visited nodes equals the number of unique nodes, the graph is connected
        return len(visited) == len(set(sum(([s, o] for s, p, o in graph), [])))

    def check_location_graph(self, graph):
        """Check the semantic validity of graphs."""
        if len(graph) != self.num_edges:
            return False

        for triple in graph:
            if not self.is_valid_triple(triple[0], triple[1], triple[2]):
                return False

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_edges", default=10, type=int, help="Number of edges (triples) in a graph")
    parser.add_argument("--train_size", default=60000, type=int, help="Size of training data split")
    parser.add_argument("--val_size", default=20000, type=int, help="Size of validation data split")
    parser.add_argument("--test_size", default=20000, type=int, help="Size of test data split")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed for the data generator")
    parser.add_argument("--dataset_name", default="syn-types", type=str, help="Name of the dataset and ZIP file")
    args = parser.parse_args()

    generator = SynTypesGenerator(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_edges=args.num_edges,
        random_seed=args.random_seed,
        dataset_name=args.dataset_name
    )

    generator.generate_and_save()
