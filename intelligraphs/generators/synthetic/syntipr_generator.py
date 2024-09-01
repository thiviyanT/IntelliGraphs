from intelligraphs.generators.synthetic.dataset_generator_base import BaseSyntheticDatasetGenerator, tqdm, random, argparse
from intelligraphs.domains.SynTIPR.entities import names, roles, years
from intelligraphs.domains.SynTIPR import relations

class SynTIPRGenerator(BaseSyntheticDatasetGenerator):
    def __init__(self, train_size=50000, val_size=10000, test_size=10000, num_edges=3, random_seed=42, dataset_name="syn-tipr"):
        super().__init__(train_size, val_size, test_size, num_edges, random_seed, dataset_name)
        # Specific data for this generator
        self.inductive_nodes = ['_academic', '_time']
        self.people = list(names)
        self.roles = list(roles)
        self.years = list(years)
        self.relations = list(relations)
        self.duration = {
            'masters researcher': [1, 2],
            'phd researcher': [3, 6],
            'post doctoral researcher': [1, 3],
            'assistant professor': [3, 5],
            'professor': [5, 30]
        }
        self.entities = self.inductive_nodes + self.people + self.roles + self.years

    def sample_synthetic_data(self, num_graphs=5):
        """Generate synthetic data specific to this generator."""
        random.shuffle(self.entities)
        triples = set()

        with tqdm(total=num_graphs) as pbar:
            while len(triples) < num_graphs:
                subgraph = []
                x0 = self.inductive_nodes[0]
                name_idx = random.randint(0, len(self.people) - 1)
                subgraph.append((x0, 'has_name', self.people[name_idx]))

                role_idx = random.randint(0, len(self.roles) - 1)
                subgraph.append((x0, 'has_role', self.roles[role_idx]))

                duration = self.duration[self.roles[role_idx]]
                duration = random.randint(duration[0], duration[1])

                x1 = self.inductive_nodes[1]
                subgraph.append((x0, 'has_time', x1))

                end_year_idx = random.randint(30, len(self.years) - 1)
                start_year_idx = end_year_idx - duration

                subgraph.append((x1, 'start_year', self.years[start_year_idx]))
                subgraph.append((x1, 'end_year', self.years[end_year_idx]))

                if self.is_valid_graph(subgraph):
                    subgraph = tuple(subgraph)
                    triples.add(subgraph)
                    pbar.update(1)

        return list(triples)

    def is_valid_graph(self, graph):
        """Check the validity of a graph."""
        v = dict()

        if len(graph) != 5:
            return False

        for triple in graph:
            s, p, o = triple
            if (s, p) in v:
                return False
            v[(s, p)] = o

            if not self.is_valid_triple(s, p, o):
                return False

        if not set([i for k, i in v.keys()]) == set(self.relations):
            return False

        if not v[('_time', 'end_year')] >= v[('_time', 'start_year')]:
            return False

        if not self.check_graph_pattern(v):
            return False

        return True

    def is_valid_triple(self, s, p, o):
        """Check if a triple is semantically valid."""
        if p == 'has_role' and o not in self.roles:
            return False
        if p == 'has_name' and o not in self.people:
            return False
        if p == 'start_year' and o not in self.years:
            return False
        if p == 'end_year' and o not in self.years:
            return False
        return True

    def check_graph_pattern(self, graph):
        """Check if a graph follows the correct pattern."""
        if ('_academic', 'has_name') not in graph or graph[('_academic', 'has_name')] not in self.people:
            return False
        if ('_academic', 'has_role') not in graph or graph[('_academic', 'has_role')] not in self.roles:
            return False
        if ('_academic', 'has_time') not in graph or graph[('_academic', 'has_time')] != '_time':
            return False
        if ('_time', 'start_year') not in graph or graph[('_time', 'start_year')] not in self.years:
            return False
        if ('_time', 'end_year') not in graph or graph[('_time', 'end_year')] not in self.years:
            return False
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", default=50000, type=int, help="Size of training data split")
    parser.add_argument("--val_size", default=10000, type=int, help="Size of validation data split")
    parser.add_argument("--test_size", default=10000, type=int, help="Size of test data split")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed for the data generator")
    parser.add_argument("--dataset_name", default="syn-tipr", type=str, help="Name of the dataset and ZIP file")
    args = parser.parse_args()

    generator = SynTIPRGenerator(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
        dataset_name=args.dataset_name
    )

    generator.generate_and_save()
