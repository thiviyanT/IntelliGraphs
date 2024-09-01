from intelligraphs.generators.synthetic.dataset_generator_base import BaseSyntheticDatasetGenerator, tqdm, random, argparse
from intelligraphs.domains.SynPaths.entities import dutch_cities
from intelligraphs.domains.SynPaths import relations

class SynPathsGenerator(BaseSyntheticDatasetGenerator):
    def __init__(self, train_size=60000, val_size=20000, test_size=20000, num_edges=3, random_seed=42, dataset_name="syn-paths"):
        super().__init__(train_size, val_size, test_size, num_edges, random_seed, dataset_name)
        # Specific data for this generator
        self.entities = list(dutch_cities)
        self.relations = list(relations)

    def sample_synthetic_data(self, num_graphs=5):
        """Generate synthetic data specific to this generator."""
        num_entities = len(self.entities)
        num_relations = len(self.relations)
        triples = set()

        with tqdm(total=num_graphs) as pbar:
            while len(triples) < num_graphs:
                subgraph = []
                selected_node = None
                _nodes = list(range(num_entities))

                relations = self.relations

                # Randomise the order of relations so that models cannot learn a shortcut
                random.shuffle(relations)

                for relation in relations:
                    if selected_node is None:
                        s = _nodes[random.randint(0, len(_nodes) - 1)]
                        _nodes.remove(s)
                        o = _nodes[random.randint(0, len(_nodes) - 1)]
                        _nodes.remove(o)
                        selected_node = o
                    else:
                        s = selected_node
                        o = _nodes[random.randint(0, len(_nodes) - 1)]
                        _nodes.remove(o)
                        selected_node = o
                    subgraph.append((self.entities[s], relation, self.entities[o]))

                if self.check_pathgraph(subgraph):
                    subgraph = tuple(subgraph)
                    triples.add(subgraph)
                    pbar.update(1)

        return list(triples)

    def check_pathgraph(self, graph):
        """Check for path graphs. Acyclic graphs are not counted as paths."""
        leaf_nodes = self.check_leaf_nodes(graph)
        if len(leaf_nodes) != 2:
            return False

        return self.follow_direction(graph, start_node=leaf_nodes[0]) or self.follow_direction(graph, start_node=leaf_nodes[1])

    def check_leaf_nodes(self, graph):
        """Employs a simple and cheap heuristic to count leaf nodes."""
        subjects = list()
        objects = list()

        for (s, p, o) in graph:
            subjects.append(s)
            objects.append(o)

        leaf_nodes = list(set(subjects) - set(objects)) + list(set(objects) - set(subjects))
        return leaf_nodes

    def follow_direction(self, graph, start_node):
        """Checks if path graph follows edges."""
        edge_connections = dict()
        for triple in graph:
            (source, _, target) = triple
            if source in edge_connections:
                return False
            edge_connections[source] = target

        i = start_node
        for _ in range(len(edge_connections)):
            f = edge_connections[i]
            edge_connections.pop(i)
            i = f

        return len(edge_connections) == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_edges", default=3, type=int, help="Number of edges (triples) in a graph")
    parser.add_argument("--train_size", default=60000, type=int, help="Size of training data split")
    parser.add_argument("--val_size", default=20000, type=int, help="Size of validation data split")
    parser.add_argument("--test_size", default=20000, type=int, help="Size of test data split")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed for the data generator")
    args = parser.parse_args()

    generator = SynPathsGenerator(
        num_edges=args.num_edges,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.random_seed
    )

    generator.generate_and_save()
