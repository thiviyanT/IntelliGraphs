from graphviz import Digraph, nohtml
from typing import List, Any, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import io
from tqdm import tqdm
from PIL import Image
from intelligraphs.errors import DataError
import warnings


class ConstraintVerifier:
    # Base class definition
    RULES = {}  # A dictionary of rules to check against graphs. This will be fined within the subclass.
    DOMAIN = {}  # A dictionary of entities that the graph can take. This will be fined within the subclass.

    def __init__(self, short_circuit: bool = True):
        """
        Initialize the GraphChecker with rules.

        Parameters:
        - rules: A dictionary of rules to check against the graph.
        - domain: A dictionary of entities that the graph can take.
        - short_circuit: If True, stop checking further rules once a rule fails.
        """
        self.rules = self.RULES
        self.domain = self.DOMAIN
        self.short_circuit = short_circuit
        self.axis_font_size = 20
        self.title_font_size = 30

        if not self.short_circuit:
            warnings.warn(
                "Short-circuiting is disabled. This means all rules will be checked for every graph, even if a rule fails early. "
                "This can significantly increase computation time and reduce efficiency, especially for large datasets. "
                "Disabling short-circuiting is primarily intended for debugging and testing purposes. ",
                UserWarning
            )

    def print_rules(self):
        """
        Print all the rules and their descriptions in a clean and organized way.
        """
        if not self.rules or len(self.rules) == 0:
            print("No rules are defined in this verifier.")
            return

        print("List of Rules and Descriptions:")
        for rule_id, rule_info in self.rules.items():
            print(f"\t-> Rule {rule_id}:")
            print(f"\t   FOL: {rule_info.get('FOL', 'N/A')}")
            print(f"\t   Description: {rule_info.get('description', 'No description provided.')}")

    def verify_dataset(self, dataset_labeled: list, dataset_name: str):
        """
        Verify a labeled dataset and raise a DataError if violations are found.

        Parameters:
        - dataset_labeled: List of labeled subgraphs.
        - dataset_name: Name of the dataset (for informative error messages).
        """
        for graph in tqdm(dataset_labeled, desc=f"Checking {dataset_name} Graphs"):
            violations = self.evaluate_graph(graph, verbose=False)
            if violations and len(violations) > 0:
                for triple in graph:
                    print(triple)
                # print(violations)
                message = self.create_violation_message(violations)
                raise DataError(f"Violations found in a graph from the {dataset_name} dataset: {message}")

    @staticmethod
    def _render_dot_to_image(dot):
        """
        Renders a graphviz.Digraph object to a PIL image.

        Parameters:
        - dot: The graphviz.Digraph object to render.

        Returns:
        - A PIL image.
        """
        png_data = dot.pipe(format='png')
        image = Image.open(io.BytesIO(png_data))
        return image

    @staticmethod
    def plot_graph_lengths_distribution(dataset: List[List[Tuple[str]]],
                                        save_path: str = 'graph_lengths_distribution.pdf') -> None:
        """
        Counts the graph lengths, plots a bar chart representing the distribution of graph lengths in the dataset, and saves it to a file.

        Parameters:
        - dataset: The dataset containing multiple graphs.
        - save_path: The path where the plot will be saved. Default is 'graph_lengths_distribution.pdf'.
        """
        plt.style.use('science')

        # Count graph lengths
        graph_lengths = defaultdict(int)
        for graph in dataset:
            graph_lengths[len(graph)] += 1

        # Convert to lists for plotting
        lengths = list(graph_lengths.keys())
        counts = list(graph_lengths.values())

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x=lengths, y=counts, palette="viridis")
        plt.xlabel('Graph Length')
        plt.ylabel('Number of Graphs')
        plt.title('Distribution of Graph Lengths')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def read_triples(file, split_tab: bool = False) -> List[List[str]]:
        """
        Read triples from a file.

        Parameters:
        - file: The file containing the triples
        - split_tab: If True, split the triples by tab. Default is False.

        Returns:
        - A list of triples
        """
        with open(file, 'r') as f:
            if split_tab:
                return [line.replace('\n', '').split('\t') for line in f]
            else:
                return [line.split() for line in f]

    @staticmethod
    def ids_to_labels(graphs: List[List[Tuple[int, int, int]]],
                      entity_labels: Dict[int, str],
                      relation_labels: Dict[int, str]) -> List[List[Tuple[str, str, str]]]:
        """
        Convert numeric IDs in graphs to their corresponding labels.

        This function processes a list of graphs where each graph is represented as a list of triples.
        Each triple consists of two entity IDs and a relation ID. The function replaces the IDs with
        their corresponding labels from the provided entity and relation label dictionaries.

        Parameters:
        - graphs: A list of graphs, where each graph is a list of triples (source entity, relation, target entity).
        - entity_labels: A dictionary mapping entity IDs to their corresponding labels.
        - relation_labels: A dictionary mapping relation IDs to their corresponding labels.

        Returns:
        - A list of graphs with labels instead of IDs.
        """
        return [
            [
                (entity_labels[s], relation_labels[p], entity_labels[o])
                for s, p, o in graph
            ]
            for graph in tqdm(graphs, desc="Processing graphs")
        ]

    @staticmethod
    def split_graphs(input_graphs: List[Tuple[str, str, str]]) -> List[List[Tuple[str, str, str]]]:
        """
        Split the graphs into groups of graphs.

        Parameters:
        - input_graphs: The list of graphs

        Returns:
        - A list of graphs
        """
        grouped_graphs = list()
        current_graph_group = list()

        for graph in input_graphs:
            if graph != ['']:
                current_graph_group.append(graph)
            else:
                grouped_graphs.append(current_graph_group)
                current_graph_group = list()

        return grouped_graphs

    @staticmethod
    def int_type_conversion(graphs: List[List[Tuple[str, str, str]]]) -> list[list[list[int | str]]]:
        """
        Converts the string type to int type.

        Parameters:
        - graphs: The list of graphs

        Returns:
        - A list of graphs
        """
        processed_graphs = [
            [
                [int(element) if element.isdigit() else element for element in triple]
                for triple in graph
            ]
            for graph in graphs
        ]

        return processed_graphs

    @staticmethod
    def _compute_correlation(data: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix for the given data, handling rows with zero standard deviation.

        Parameters:
        - data: The data to compute the correlation matrix for.

        Returns:
        - The correlation matrix.
        """
        correlation = np.corrcoef(data, rowvar=False)
        correlation[np.isnan(correlation)] = 0  # Replace nan values with 0
        return correlation

    def check_rules_for_dataset(self, graphs: List[List[Tuple[str, str, str]]]) -> list[list[str]]:
        """
        Check the given dataset against the rules.

        Parameters:
        - dataset: The dataset containing multiple graphs.

        Returns:
        - A list of lists where each inner list contains the rule violations for a graph.
        """
        return [
            [violations for _, violations in self.check_rules_for_graph(graph)]
            for graph in graphs
        ]

    def check_rules_for_graph(self, graph: List[Tuple[str, str, str]]) -> List[Tuple[int, str]]:
        """
        Check the given graph against the rules.

        Parameters:
        - graph: The graph to check.

        Returns:
        - A list of tuples where each tuple contains a result (1 or 0) and a failure message (if applicable).
        """

        rule_results = []
        for rule_id, rules in self.rules.items():
            failure_message = rules["failure_message"] if rules["violations"] else ""
            violations = rules["violations"](graph, self.domain) if rules["violations"] else set()
            rule_results.append((failure_message, violations))
            if self.short_circuit and len(violations) > 0:
                break

        return rule_results

    def evaluate_graph(self, graph: Any, verbose: bool = False) -> list[tuple[int, str]]:
        """
        Evaluate a graph and print it with any rule violations.

        Parameters:
        - graph: The graph to evaluate.
        """
        # Assert to make sure the graph is a list
        assert isinstance(graph, list), "The input graph must be a list of triples."

        results = self.check_rules_for_graph(graph)
        failures = [(msg, violations) for msg, violations in results if len(violations) > 0]


        if verbose:
            print(graph)
            print("Graph:")
            for triple in graph:
                print(f"  {triple[0]} -- {triple[1]} --> {triple[2]}")
            if failures:
                print("\nFailure Reasons:")
                for msg, violations in failures:
                    print(f"  - {msg}")
                    if violations:  # If there are any violation triples
                        print("    Violations:")
                        for v in violations:
                            print(f"      {v[0]} -- {v[1]} --> {v[2]}")
            else:
                print("\nGraph passes all rules!")
        return failures

    @staticmethod
    def create_violation_message(violations):
        messages = []
        for failure_message, violation_set in violations:
            if violation_set:
                violation_details = ', '.join(
                    [f"({s}, {p}, {o})" for s, p, o in violation_set]
                )
                message = f"\n\t- {failure_message} The following violation(s) were found: {violation_details}."
                messages.append(message)
        return '\n'.join(messages)

    def visualize_graph(self, graph: Any, show_violations: bool = False) -> None:
        """
        Visualize the given graph using Graphviz and optionally show rule violations using matplotlib.

        Parameters:
        - graph: The graph to visualize.
        - show_violations: If True, display rule violations above the graph.
        """
        dot = Digraph(comment="Graph Visualization")
        for triple in graph:
            subject, relation, obj = triple
            subject = nohtml(subject)
            obj = nohtml(obj)
            dot.node(subject, subject)
            dot.node(obj, obj)
            dot.edge(subject, obj, label=relation)

        img = self._render_dot_to_image(dot)

        ax = plt.gca()
        ax.imshow(img)
        ax.axis('off')

        if show_violations:
            results = self.check_rules_for_graph(graph)
            violations = [msg for res, msg, vio in results if not res]

            if violations:
                all_msgs = "\n".join(violations)
                ax.set_title(all_msgs, color="red", fontsize=8)

    def sample_and_visualize(self,
                             list_of_graphs: List[Any],
                             show_violations: bool = False,
                             num_cols: int = 3,
                             num_rows: int = 5,
                             filename: str = None) -> None:
        """
        Sample a specified graphs from the list and visualize them.

        Parameters:
        - list_of_graphs: The list of graphs to sample from.
        - show_violations: If True, display rule violations above the graph.
        """
        plt.style.use('science')
        sampled_graphs = random.sample(list_of_graphs, num_cols * num_rows)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 16))

        for i in range(num_rows):
            for j in range(num_cols):
                plt.sca(axes[i, j])
                self.visualize_graph(sampled_graphs[i * num_cols + j], show_violations=show_violations)

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_valid_graphs(self, list_of_graphs: List[Any], num_to_sample: int = 20,
                          show_violations: bool = False, filename: str = 'valid_graphs.pdf') -> None:
        """
        Sample and visualize valid graphs.

        Parameters:
        - list_of_graphs: The list of graphs to sample from.
        - num_to_sample: The number of valid graphs to sample and visualize.
        - show_violations: If True, display rule violations below the graph.
        """
        # Filter for valid graphs
        valid_graphs = [graph for graph in list_of_graphs if
                        all(res == 1 for res, _ in self.check_rules_for_graph(graph))]

        if len(valid_graphs) > num_to_sample:
            valid_graphs = random.sample(valid_graphs, num_to_sample)

        self.sample_and_visualize(valid_graphs, show_violations=show_violations, filename=filename)

    def plot_invalid_graphs(self, list_of_graphs: List[Any], num_to_sample: int = 20,
                            show_violations: bool = True, filename: str = 'invalid_graphs.pdf') -> None:
        """
        Sample and visualize invalid graphs.

        Parameters:
        - list_of_graphs: The list of graphs to sample from.
        - num_to_sample: The number of invalid graphs to sample and visualize.
        - show_violations: If True, display rule violations below the graph.
        """
        # Filter for invalid graphs
        invalid_graphs = [graph for graph in list_of_graphs if
                          any(res == 0 for res, _, _ in self.check_rules_for_graph(graph))]

        if len(invalid_graphs) > num_to_sample:
            invalid_graphs = random.sample(invalid_graphs, num_to_sample)

        self.sample_and_visualize(invalid_graphs, show_violations=show_violations, filename=filename)

    def plot_violations_heatmap(self, dataset: List[List[Tuple[str]]], show_functions: bool = True) -> None:
        """
        Plots a heatmap with function and rule violations.

        Parameters:
        - dataset: The dataset containing multiple graphs.
        - show_functions: If True, show the function violations heatmap. Default is True.
        """
        plt.style.use('science')

        # Get the results for all graphs in the dataset
        results_dataset = [self.check_rules_for_graph(graph) for graph in dataset]

        # Order the dataset by number of violated rules
        results_dataset.sort(key=lambda results: sum(1 for res, _ in results if res == 0), reverse=True)

        # Extract just the results (ignoring failure messages) for plotting
        num_rules = len(self.rules)
        function_violations_data = [[res for res, _ in results] + [1] * (num_rules - len(results)) for results in
                                    results_dataset]
        function_violations = np.array(function_violations_data, dtype=int).T

        # Assuming every rule is represented by just one function
        rule_violations = function_violations

        # Compute correlation between rule violations
        correlation_matrix = self._compute_correlation(rule_violations.T)

        # Combining the plots into one figure
        if show_functions:

            # Combining the three plots into one figure
            fig_combined, axes_combined = plt.subplots(nrows=3, ncols=1, figsize=(20, 27))

            # Top heatmap with functions
            sns.heatmap(function_violations, cmap=['white', 'green'], cbar=False,
                        yticklabels=range(1, rule_violations.shape[0] + 1), xticklabels=[], ax=axes_combined[0])
            axes_combined[0].set_ylabel("Functions", fontsize=self.axis_font_size)
            axes_combined[0].set_xlabel("Graphs", fontsize=self.axis_font_size)
            axes_combined[0].set_title("Function Violations", fontsize=self.title_font_size)
        else:
            fig_combined, axes_combined = plt.subplots(nrows=2, ncols=1, figsize=(20, 18))

        # Middle heatmap with rules
        sns.heatmap(rule_violations, cmap=['white', 'green'], cbar=False,
                    yticklabels=range(1, rule_violations.shape[0] + 1), xticklabels=[], ax=axes_combined[-2])
        axes_combined[-2].set_ylabel("Rules", fontsize=self.axis_font_size)
        axes_combined[-2].set_xlabel("Graphs", fontsize=self.axis_font_size)
        axes_combined[-2].set_title("Rules Violations", fontsize=self.title_font_size)

        # Bottom heatmap for correlation between rules
        sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True,
                    yticklabels=range(1, correlation_matrix.shape[0] + 1),
                    xticklabels=range(1, correlation_matrix.shape[0] + 1), ax=axes_combined[-1])
        axes_combined[-1].set_ylabel("Rules", fontsize=self.axis_font_size)
        axes_combined[-1].set_xlabel("Rules", fontsize=self.axis_font_size)
        axes_combined[-1].set_title("Correlation between Rules", fontsize=self.title_font_size)
        # Set the aspect ratio for the correlation heatmap to be equal (to make it square)
        axes_combined[-1].set_aspect('equal')

        # fig_combined.tight_layout()

        # Display rule descriptions below the plots
        # rule_texts = [rule_data["description"] for rule_data in self.RULES.values()]
        # for idx, rule_text in enumerate(rule_texts, start=1):
        #     plt.text(0, -idx * 0.02, f"Rule {idx}: {rule_text}", transform=fig_combined.transFigure, fontsize=20)
        #
        # plt.subplots_adjust(bottom=0.0 + len(rule_texts) * 0.02)  # Adjust the bottom space to fit the rule descriptions

        plt.show()
        fig_combined.savefig(f"{self.__class__.__name__}-Heatmap.pdf")

    def count_rule_violations(self, dataset: List[List[Tuple[str]]]) -> Dict[str, int]:
        """
        Count the number of times each rule is violated across all graphs in the dataset.

        Parameters:
        - dataset: The dataset containing multiple graphs.

        Returns:
        - A dictionary where keys are the failure messages and values are the counts of violations.
        """
        violation_counts = defaultdict(int)

        for graph in dataset:
            results = self.check_rules_for_graph(graph)
            for _, failure_message in results:
                if failure_message:  # If there's a violation
                    violation_counts[failure_message] += 1

        return violation_counts

    def plot_top_k_violations(self, dataset: List[List[Tuple[str]]], k: int = None) -> None:
        """
        Plots the top k rule violations in a bar chart.

        Parameters:
        - dataset: The dataset containing multiple graphs.
        - k: The number of top violations to plot.
        """

        # Use the science style for plots
        plt.style.use('science')

        violation_counts = self.count_rule_violations(dataset)

        # If k is not provided, plot all violations
        if k is None:
            k = len(violation_counts)

        # Sort violations by count and take top k
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:k]

        # Extract data for plotting
        messages = [item[0] for item in sorted_violations]
        counts = [item[1] for item in sorted_violations]

        # Plot the bar chart
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        bars = sns.barplot(x=messages, y=counts, palette="viridis", ax=axs[0])
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
        axs[0].set_ylabel('Violation Frequency (absolute)', fontsize=self.axis_font_size)
        axs[0].set_yticks(np.arange(0, max(counts) + 1, 500))

        if k == len(violation_counts):
            axs[0].set_title(f'All Rule Violations')
        else:
            axs[0].set_title(f'Top {k} Rule Violations')

        # Add violation count labels above each bar
        for bar in bars.patches:
            height = bar.get_height()
            axs[0].text(bar.get_x() + bar.get_width() / 2., height + 5,
                        '%d' % int(height), ha='center', va='bottom')

        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

        fig.savefig(f"{self.__class__.__name__}-{k}Violations.pdf")

    def plot_all_rules(self, dataset: List[List[Tuple[str]]], plot_satisfaction: bool = False) -> None:
        """
        Plots all the rules, highlighting their violation counts in the dataset.
        If plot_satisfaction is True, it plots rule satisfaction counts.

        Parameters:
        - dataset: The dataset containing multiple graphs.
        - plot_satisfaction: Flag to plot rule satisfaction instead of violations. Default is False.
        """

        # Use the science style for plots
        plt.style.use('science')

        # Get violation counts
        violation_counts = self.count_rule_violations(dataset)

        # Ensure all rules are present in the counts (even if they're not violated)
        total_graphs = len(dataset)
        for rule, rule_data in self.rules.items():
            if rule_data["failure_message"] not in violation_counts:
                violation_counts[rule_data["failure_message"]] = 0

            # If we're plotting satisfaction, convert violation count to satisfaction count
            if plot_satisfaction:
                violation_counts[rule_data["failure_message"]] = total_graphs - violation_counts[
                    rule_data["failure_message"]]

            # Sort rules by their counts for better visualization

        # Sort rules by their violation counts for better visualization
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)

        # Extract data for plotting
        messages = [item[0] for item in sorted_violations]
        counts = [item[1] for item in sorted_violations]

        # Plot the bar chart
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        bars = sns.barplot(x=messages, y=counts, palette="viridis", ax=axs[0])
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
        axs[0].set_ylabel('Frequency (absolute)', fontsize=self.axis_font_size)
        axs[0].set_yticks(np.arange(0, max(counts) + 1, 500))
        title = f'Rule Satisfactions' if plot_satisfaction else f'Rule Violations'
        axs[0].set_title(title, fontsize=self.title_font_size)

        # Add violation count labels above each bar
        for bar in bars.patches:
            height = bar.get_height()
            axs[0].text(bar.get_x() + bar.get_width() / 2., height + 5,
                        '%d' % int(height), ha='center', va='bottom')

        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

        filename = f"{self.__class__.__name__}-AllRulesSatisfactions.pdf" \
            if plot_satisfaction else f"{self.__class__.__name__}-AllRulesViolations.pdf"
        fig.savefig(filename)

    # def verify_dataset(self, data_directory: str = None) -> bool:
    #     """
    #     Check all the graphs in the dataset (train, valid & test split). Raises an error, if any.
    #     """
    #     assert data_directory is not None, 'data directory must be provided'
    #     assert not self.short_circuit, 'short circuit must be disabled for verifying dataset against all the rules'
    #
    #     if not os.path.isdir(data_directory):
    #         raise FileNotFoundError('Data path does not exist!')
    #
    #     print('Reading train, validation and test data from file...')
    #     train_file = f'{data_directory}/train.txt'
    #     val_file = f'{data_directory}/valid.txt'
    #     test_file = f'{data_directory}/test.txt'
    #
    #     train = self.read_triples(train_file, split_tab=True)
    #     val = self.read_triples(val_file, split_tab=True)
    #     test = self.read_triples(test_file, split_tab=True)
    #
    #     train = self.split_graphs(train)
    #     val = self.split_graphs(val)
    #     test = self.split_graphs(test)
    #
    #     train = self.int_type_conversion(train)
    #     val = self.int_type_conversion(val)
    #     test = self.int_type_conversion(test)
    #
    #     print('Checking all data against all rules...')
    #     for graph in tqdm(train + val + test):
    #         rst = self.check_rules_for_graph(graph)
    #         failures = [(msg, violations) for msg, violations in rst if len(violations) > 0]
    #         if failures:
    #             print(graph)
    #             print("\nFailure Reasons:")
    #             for msg, violations in failures:
    #                 print(f"  - {msg}")
    #                 if violations:  # If there are any violation triples
    #                     print("    Violations:")
    #                     for v in violations:
    #                         print(f"      {v[0]} -- {v[1]} --> {v[2]}")
    #             raise TypeError
    #     print('All data passed all rules!')
    #     return True

    def evaluate_model_output(self, dataset: List[List[Tuple[str, str, str]]]) -> Dict[str, float]:
        """
        Generate a summary of the semantic check.

        Parameters:
        - dataset: The dataset containing multiple graphs.

        Returns:
        - A dictionary containing the "% Validity" for both checks.
        """
        # Check the entire dataset against the rules
        results = self.check_rules_for_dataset(dataset)

        # Mark a graph as valid if it has no violations
        valid_graphs = sum(1 for result in results if all(len(violations) == 0 for violations in result))
        total_graphs = len(dataset)

        # Compute the "% Validity" for both checks
        validity = (valid_graphs / total_graphs) * 100

        return {
            "Consistency Score (%) (complete graph)": round(validity, 2),
            "Consistency Score (%) (partial graph)": None,
        }
