from intelligraphs.verifier.constraint_verifier_base import ConstraintVerifier
from intelligraphs.domains.SynPaths.entities import dutch_cities
from intelligraphs.domains.SynPaths.relations import relations

class SynPathsVerifier(ConstraintVerifier):
    DOMAIN = {
        "dutch_cities": dutch_cities
    }

    RELATIONS = relations

    @staticmethod
    def rule_1_is_path_graph(graph, domain):
        degrees = {}

        for u, _, v in graph:
            degrees[u] = degrees.get(u, {"in": 0, "out": 0})
            degrees[v] = degrees.get(v, {"in": 0, "out": 0})

            degrees[u]["out"] += 1
            degrees[v]["in"] += 1

        start_vertices = sum(1 for v in degrees.values() if v["in"] == 0 and v["out"] == 1)
        end_vertices = sum(1 for v in degrees.values() if v["in"] == 1 and v["out"] == 0)
        middle_vertices = sum(1 for v in degrees.values() if v["in"] == 1 and v["out"] == 1)

        return graph if not (start_vertices == 1 and end_vertices == 1 and middle_vertices == len(degrees) - 2 and len(graph) == len(degrees) - 1) else set()

    @staticmethod
    def rule_3_has_root_node(graph, domain):
        degrees = {}
        for u, _, v in graph:
            degrees[u] = degrees.get(u, {"in": 0, "out": 0})
            degrees[v] = degrees.get(v, {"in": 0, "out": 0})

            degrees[u]["out"] += 1
            degrees[v]["in"] += 1

        return graph if not any(v["in"] == 0 and v["out"] == 1 for v in degrees.values()) else set()

    @staticmethod
    def rule_4_single_root_node(graph, domain):
        root_nodes = SynPathsVerifier.find_root_nodes(graph)
        return graph if len(root_nodes) != 1 else set()

    @staticmethod
    def rule_6_no_self_loops(graph, domain):
        return {(s, p, o) for s, p, o in graph if s == o}

    @staticmethod
    def rule_7_validate_root_connections(graph, domain):
        root_nodes = SynPathsVerifier.find_root_nodes(graph)

        for node in {s for s, p, o in graph} | {o for s, p, o in graph}:
            if not any(root in SynPathsVerifier.reverse_traverse(graph, node) for root in root_nodes):
                if len(root_nodes) == 1:
                    return {}

        return graph

    @staticmethod
    def rule_8_check_unique_parent_per_target(graph, domain):
        target_to_sources = {}

        for subject, _, obj in graph:
            if obj not in target_to_sources:
                target_to_sources[obj] = set()
            target_to_sources[obj].add(subject)

        for sources in target_to_sources.values():
            if len(sources) > 1:
                return graph

        return set()

    @staticmethod
    def rule_9_check_unique_target_per_source(graph, domain):
        source_to_targets = {}

        for subject, _, obj in graph:
            if subject not in source_to_targets:
                source_to_targets[subject] = set()
            source_to_targets[subject].add(obj)

        for targets in source_to_targets.values():
            if len(targets) > 1:
                return graph

        return set()

    @staticmethod
    def rule_10_check_edge_correspondence(graph, domain):
        return graph if not all(predicate in {p for _, p, _ in graph} for predicate in ["cycle_to", "drive_to", "train_to"]) else set()

    @staticmethod
    def rule_11_check_single_occurrence_cycle_to(graph, domain):
        return SynPathsVerifier.check_single_occurrence(graph, "cycle_to")

    @staticmethod
    def rule_12_check_single_occurrence_drive_to(graph, domain):
        return SynPathsVerifier.check_single_occurrence(graph, "drive_to")

    @staticmethod
    def rule_13_check_single_occurrence_train_to(graph, domain):
        return SynPathsVerifier.check_single_occurrence(graph, "train_to")

    @staticmethod
    def rule_14_check_triple_count(graph, domain, expected_count):
        return graph if len(graph) != expected_count else set()

    @staticmethod
    def find_root_nodes(graph):
        in_degree = {}
        out_degree = {}

        # Calculate in-degree and out-degree for each node
        for subject, predicate, obj in graph:
            out_degree[subject] = out_degree.get(subject, 0) + 1
            in_degree[obj] = in_degree.get(obj, 0) + 1

        # A root node should have an in-degree of 0 and exactly one outgoing edge
        root_nodes = {node for node in out_degree if in_degree.get(node, 0) == 0 and out_degree[node] == 1}

        return root_nodes

    @staticmethod
    def reverse_traverse(graph, start_node, visited=None):
        if visited is None:
            visited = set()

        visited.add(start_node)

        # Traverse the graph in reverse
        for subject, predicate, obj in graph:
            if predicate == "edge" and obj == start_node and subject not in visited:
                SynPathsVerifier.reverse_traverse(graph, subject, visited)
        return visited

    @staticmethod
    def check_single_occurrence(graph, predicate):
        occurrences = [(s, p, o) for s, p, o in graph if p == predicate]
        if len(occurrences) > 1:
            return set(occurrences)
        elif len(occurrences) == 0:
            return {("<missing>", predicate, "<missing>")}
        return set()


    RULES = {
        1: {
            "FOL": "∀x, y, z: connected(x, y) ∧ connected(y, z) ⇒ connected(x, z)",
            "violations": lambda graph, domain: SynPathsVerifier.rule_1_is_path_graph(graph, domain),
            "description": "Ensures transitivity. If x is connected to y, and y is connected to z, then x should be connected to z.",
            "failure_message": "Rule 1: The graph is missing a connection where x should be connected to z through y.",
        },
        2: {
            "FOL": "∀x, y: edge(x, y) ⇒ connected(x, y)",
            "violations": lambda graph, domain: {},
            "description": "If there's an edge between two nodes x and y, then x should be connected to y.",
            "failure_message": "Rule 2: The edge between x and y does not establish a connection, which should never happen.",
        },
        3: {
            "FOL": "∃x: root(x)",
            "violations": lambda graph, domain: SynPathsVerifier.rule_3_has_root_node(graph, domain),
            "description": "There must be at least one root node in the graph.",
            "failure_message": "Rule 3: No root node was found in the graph.",
        },
        4: {
            "FOL": "∀x, y: root(x) ∧ root(y) ⇒ x=y",
            "violations": lambda graph, domain: SynPathsVerifier.rule_4_single_root_node(graph, domain),
            "description": "There can only be one root node in the graph.",
            "failure_message": "Rule 4: Multiple root nodes were detected, but there should only be one.",
        },
        5: {
            "FOL": "∀x: root(x) ⇔ ∀y: ¬edge(y, x)",
            "violations": lambda graph, domain: {},
            "description": "A root node is defined as having no incoming edges.",
            "failure_message": "Rule 5: The supposed root node has incoming edges, which should not be the case.",
        },
        6: {
            "FOL": "∀x, y: connected(x, y) ⇒ x ≠ y",
            "violations": lambda graph, domain: SynPathsVerifier.rule_6_no_self_loops(graph, domain),
            "description": "No node should be connected to itself, preventing self-loops in the graph.",
            "failure_message": "Rule 6: A self-loop is detected where a node is incorrectly connected to itself.",
        },
        7: {
            "FOL": "∀x: root(x) ⇒ ∀y: (connected(x, y) or x=y)",
            "violations": lambda graph, domain: SynPathsVerifier.rule_7_validate_root_connections(graph, domain),
            "description": "Ensures that all nodes are either connected to a root node or are a root node themselves.",
            "failure_message": "Rule 7: There is at least one node that is not connected to any root node nor is a root itself.",
        },
        8: {
            "FOL": "∀x, y, z: edge(y, x) ∧ edge(z, x) ⇒ y=z",
            "violations": lambda graph, domain: SynPathsVerifier.rule_8_check_unique_parent_per_target(graph, domain),
            "description": "No node should have multiple different nodes pointing to it (only one parent node is allowed).",
            "failure_message": "Rule 8: A node has multiple distinct parents, which is not allowed.",
        },
        9: {
            "FOL": "∀x, y, z: edge(x, y) ∧ edge(x, z) ⇒ y=z",
            "violations": lambda graph, domain: SynPathsVerifier.rule_9_check_unique_target_per_source(graph, domain),
            "description": "A node should not have edges pointing to two different nodes.",
            "failure_message": "Rule 9: A node has multiple outgoing edges to different nodes, which should not happen.",
        },
        10: {
            "FOL": "∀x, y: edge(x, y) ⇔ cycle_to(x, y) ∨ drive_to(x, y) ∨ train_to(x, y)",
            "violations": lambda graph, domain: SynPathsVerifier.rule_10_check_edge_correspondence(graph, domain),
            "description": "Every edge should correspond to one of the predicates: cycle_to, drive_to, or train_to.",
            "failure_message": "Rule 10: An edge does not correspond to any of the expected predicates (cycle_to, drive_to, or train_to).",
        },
        11: {
            "FOL": "∃x( cycle_to(x) ∧ ¬∃y( cycle_to(y) ∧ y ≠ x))",
            "violations": lambda graph, domain: SynPathsVerifier.rule_11_check_single_occurrence_cycle_to(graph, domain),
            "description": "The 'cycle_to' relationship should occur exactly once in the graph.",
            "failure_message": "Rule 11: The 'cycle_to' relationship either did not occur or occurred more than once in the graph.",
        },
        12: {
            "FOL": "∃x( drive_to(x) ∧ ¬∃y( drive_to(y) ∧ y ≠ x))",
            "violations": lambda graph, domain: SynPathsVerifier.rule_12_check_single_occurrence_drive_to(graph, domain),
            "description": "The 'drive_to' relationship should occur exactly once in the graph.",
            "failure_message": "Rule 12: The 'drive_to' relationship either did not occur or occurred more than once in the graph.",
        },
        13: {
            "FOL": "∃x( train_to(x) ∧ ¬∃y( train_to(y) ∧ y ≠ x))",
            "violations": lambda graph, domain: SynPathsVerifier.rule_13_check_single_occurrence_train_to(graph, domain),
            "description": "The 'train_to' relationship should occur exactly once in the graph.",
            "failure_message": "Rule 13: The 'train_to' relationship either did not occur or occurred more than once in the graph.",
        },
        14: {
            "FOL": "count(graph) = 3",
            "violations": lambda graph, domain: SynPathsVerifier.rule_14_check_triple_count(graph, domain, expected_count=3),
            "description": "Ensures the graph contains exactly the expected number of triples.",
            "failure_message": "Rule 14: The number of triples in the graph does not match the expected count.",
        },
    }
