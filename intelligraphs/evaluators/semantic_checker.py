from tqdm import trange
from typing import List, Tuple, Dict, Union, Set, Callable, Any


class SemanticEvaluator:
    def __init__(self,
                 predicted_graphs: List[List[Tuple[str, str, Union[str, int]]]] = None,
                 ground_truth_graphs: List[List[Tuple[str, str, Union[str, int]]]] = None,
                 rule_checker: Callable[[List[Tuple[str, str, Union[str, int]]]], List[Tuple[str, List[str]]]] = None,
                 entity_labels: List[str] = None,
                 relation_labels: List[str] = None):
        """
        Initializes the SemanticEvaluator with the necessary data.
        """
        self.predicted_graphs = predicted_graphs
        self.ground_truth_graphs = ground_truth_graphs
        self.rule_checker = rule_checker
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels

        self.valid = 0  # Number of valid graphs
        self.valid_graphs = set()  # Set of valid graphs
        self.valid_novel = 0  # Number of valid graphs that are not present in the training data
        self.valid_novel_graphs = set()  # Set of valid graphs that are not present in the training data
        self.novel_graphs = 0  # Number of graphs that are not present in the training data
        self.known_graphs = 0  # Number of graphs that are present in the training data
        self.empty_graph = 0  # Number of empty graphs
        self.invalid_graphs = set()  # Set of invalid graphs

        assert self.is_labeled_graphs(self.predicted_graphs), \
            "Predicted graphs are not labeled. Semantic Evaluator cannot process unlabelled graphs."
        assert self.is_labeled_graphs(self.ground_truth_graphs), \
            "Ground truth graphs are not labeled. Semantic Evaluator cannot process unlabelled graphs."

    @staticmethod
    def is_labeled_graphs(graphs: List[List[Tuple[str, str, Union[str, int]]]]) -> bool:
        """
        Verifies that the input graphs are properly labeled. Each graph should be a list of triples,
        with each triple containing three string elements.
        """
        if not isinstance(graphs, list):
            return False
        for graph in graphs:
            if not isinstance(graph, list):
                return False
            for triple in graph:
                if not (isinstance(triple, (list, tuple)) and len(triple) == 3):
                    return False
                subject, predicate, obj = triple
                if not (isinstance(subject, str) and isinstance(predicate, str) and (isinstance(obj, str) or isinstance(obj, int))):
                    return False
        return True

    def check_novelty(self, graph: List[Tuple[str, str, Union[str, int]]]) -> bool:
        """
        Checks whether the graph is novel by comparing it against the ground truth graphs.
        """
        assert isinstance(graph, (list, tuple)), "Graph must be a list or tuple."
        assert isinstance(self.ground_truth_graphs, list), "Ground truth graphs must be a list."

        return graph not in self.ground_truth_graphs

    def is_empty_graph(self, graph: List[Tuple[str, str, Union[str, int]]]) -> bool:
        """
        Checks if the graph is empty and ensures the input type is correct.
        """
        assert isinstance(graph, (list, tuple)), "Graph must be a list or tuple."
        return len(graph) == 0

    def check_graph(self, graph: List[Tuple[str, str, Union[str, int]]]) -> bool:
        """
        Checks the semantic validity of a single graph.

        Parameters:
        - graph: The graph to check, represented as a list of triples.

        Returns:
        - True if the graph passes all rules (no violations), otherwise False.
        """
        # Assert to make sure the graph is a list of triples
        assert isinstance(graph, list), "The input graph must be a list of triples."

        # Check if the graph is empty
        if self.is_empty_graph(graph):
            self.empty_graph += 1
            return False

        # Determine if the graph is novel
        is_novel = self.check_novelty(graph)

        # Check the graph against the rules using the appropriate rule checking function
        results = self.rule_checker(graph)

        # Determine validity based on whether there are any violations
        is_valid = all(len(violations) == 0 for _, violations in results)

        # Convert each list inside the graph to a tuple to make the entire graph hashable
        graph_tuple = tuple(tuple(edge) for edge in graph)

        # Categorise the graph based on the results
        if is_novel:
            self.novel_graphs += 1
            if is_valid:
                self.valid_novel += 1
                self.valid += 1
                self.valid_novel_graphs.add(graph_tuple)
            else:
                self.invalid_graphs.add(graph_tuple)
        else:
            self.known_graphs += 1
            if is_valid:
                self.valid += 1
                self.valid_graphs.add(graph_tuple)
            else:
                self.invalid_graphs.add(graph_tuple)

        return is_valid

    def evaluate_graphs(self) -> Any:
        """
        Iterates over all graphs and checks their semantic validity.
        """

        for i in trange(len(self.predicted_graphs)):
            self.check_graph(self.predicted_graphs[i])

        pct_semantics = round((self.valid / len(self.predicted_graphs)) * 100, 2)
        pct_novel_semantics = round((self.valid_novel / len(self.predicted_graphs)) * 100, 2)
        pct_novel = round((self.novel_graphs / len(self.predicted_graphs)) * 100, 2)
        pct_known = round((self.known_graphs / len(self.predicted_graphs)) * 100, 2)
        pct_empty = round((self.empty_graph / len(self.predicted_graphs)) * 100, 2)

        self.organized_results = {
            'results': {
                'semantics': pct_semantics,
                'novel_semantics': pct_novel_semantics,
                'novel': pct_novel,
                'known': pct_known,
                'empty': pct_empty,
            },
            'graphs': {
                'valid_graphs': {
                    'count': len(self.valid_graphs),
                    'sample': list(self.valid_graphs)
                },
                'valid_novel_graphs': {
                    'count': len(self.valid_novel_graphs),
                    'sample': list(self.valid_novel_graphs)
                },
                'invalid_graphs': {
                    'count': len(self.invalid_graphs),
                    'sample': list(self.invalid_graphs)
                }
            }
        }

        return self.organized_results

    def print_results(self, include_graph_samples:bool = True, num_samples:int = 5):
        """
        Prints the organized results in a structured manner.
        """
        print('Semantic Evaluation Results:')
        for key, value in self.organized_results['results'].items():
            print(f"{key.capitalize()}: {value}%")

        if not include_graph_samples:
            return

        print(f'\nGraph Samples (Printing only {num_samples} samples):')
        for graph_type, data in self.organized_results['graphs'].items():
            print(f"\n{graph_type.replace('_', ' ').capitalize()}:")
            print(f"Count: {data['count']}")
            print(f"Sample: {data['sample'][:num_samples]}")
