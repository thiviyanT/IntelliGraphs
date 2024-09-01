from intelligraphs.verifier.constraint_verifier_base import ConstraintVerifier
from intelligraphs.domains.SynTypes.entities import countries, cities, languages
from intelligraphs.domains.SynTypes.relations import relations
from typing import List, Tuple, Set

class SynTypesVerifier(ConstraintVerifier):
    # Subclass definition for SynTypes
    DOMAIN = {
        'countries': countries,
        'languages': languages,
        'cities': cities
    }

    RELATIONS = relations

    @staticmethod
    def rule_1_validate_spoken_in(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that 'spoken_in' relationship is correct, i.e., the subject is a language and the object is a country.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if predicate == "spoken_in":
                if subject not in SynTypesVerifier.DOMAIN["languages"] or obj not in SynTypesVerifier.DOMAIN[
                    "countries"]:
                    violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_2_validate_could_be_part_of(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that 'could_be_part_of' relationship is correct, i.e., the subject is a city and the object is a country.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if predicate == "could_be_part_of":
                if subject not in SynTypesVerifier.DOMAIN["cities"] or obj not in SynTypesVerifier.DOMAIN["countries"]:
                    violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_3_validate_same_type_as(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that 'same_type_as' relationship is correct, i.e., both entities should be of the same type.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if predicate == "same_type_as":
                if not (
                        (subject in SynTypesVerifier.DOMAIN["languages"] and obj in SynTypesVerifier.DOMAIN[
                            "languages"]) or
                        (subject in SynTypesVerifier.DOMAIN["cities"] and obj in SynTypesVerifier.DOMAIN["cities"]) or
                        (subject in SynTypesVerifier.DOMAIN["countries"] and obj in SynTypesVerifier.DOMAIN[
                            "countries"])
                ):
                    violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_4_validate_exclusive_language(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that a language entity does not belong to the domain of countries or cities.
        """
        violations = set()
        other_types = ["cities", "countries"]

        for node in SynTypesVerifier.DOMAIN["languages"]:
            for other_type in other_types:
                if node in SynTypesVerifier.DOMAIN[other_type]:
                    violations.add((node, f"invalid_type_as_{other_type}", node))

        return violations

    @staticmethod
    def rule_5_validate_exclusive_country(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that a country entity does not belong to the domain of languages or cities.
        """
        violations = set()
        other_types = ["languages", "cities"]

        for node in SynTypesVerifier.DOMAIN["countries"]:
            for other_type in other_types:
                if node in SynTypesVerifier.DOMAIN[other_type]:
                    violations.add((node, f"invalid_type_as_{other_type}", node))

        return violations

    @staticmethod
    def rule_6_validate_exclusive_city(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that a city entity does not belong to the domain of languages or countries.
        """
        violations = set()
        other_types = ["languages", "countries"]

        for node in SynTypesVerifier.DOMAIN["cities"]:
            for other_type in other_types:
                if node in SynTypesVerifier.DOMAIN[other_type]:
                    violations.add((node, f"invalid_type_as_{other_type}", node))

        return violations

    @staticmethod
    def rule_7_validate_connectedness(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that the triples in the provided subgraph form a connected component.
        Returns a set of triples that are not connected.
        """
        if not graph:
            return set()  # No triples to validate

        # Create an adjacency list for the graph
        adjacency_list = {}
        for subject, _, obj in graph:
            if subject not in adjacency_list:
                adjacency_list[subject] = set()
            if obj not in adjacency_list:
                adjacency_list[obj] = set()
            adjacency_list[subject].add(obj)
            adjacency_list[obj].add(subject)

        # Perform BFS to find all connected nodes
        visited = set()
        nodes = list(adjacency_list.keys())

        def bfs(start_node):
            queue = [start_node]
            while queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    queue.extend(adjacency_list[node] - visited)

        # Start BFS from the first node
        bfs(nodes[0])

        # Identify triples where both entities are not in the visited set
        violations = set()
        for subject, predicate, obj in graph:
            if subject not in visited or obj not in visited:
                violations.add((subject, predicate, obj))

        return violations

    @staticmethod
    def rule_8_validate_allowed_relationships(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that only the specified relationships are present in the graph.
        """
        allowed_relationships = SynTypesVerifier.RELATIONS
        violations = set()
        for subject, predicate, obj in graph:
            if predicate not in allowed_relationships:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_9_validate_no_self_loops(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that there are no self-loops in the graph, i.e., no entity has a relationship with itself.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if subject == obj:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_10_check_triple_count(graph, expected_count):
        """
        Checks that the graph contains exactly the expected number of triples.
        """
        return graph if len(graph) != expected_count else set()

    RULES = {
        1: {
            "FOL": "∀x,y: spoken_in(x, y) ⇒ language(x) ∧ country(y)",
            "violations": lambda graph, domain: SynTypesVerifier.rule_1_validate_spoken_in(graph),
            "description": "Validates that the subject of 'spoken_in' is a language and the object is a country.",
            "failure_message": "Rule 1: Invalid 'spoken_in' relationship found.",
        },
        2: {
            "FOL": "∀x,y: could_be_part_of(x, y) ⇒ city(x) ∧ country(y)",
            "violations": lambda graph, domain: SynTypesVerifier.rule_2_validate_could_be_part_of(graph),
            "description": "Validates that the subject of 'could_be_part_of' is a city and the object is a country.",
            "failure_message": "Rule 2: Invalid 'could_be_part_of' relationship found.",
        },
        3: {
            "FOL": "∀x,y: same_type_as(x, y) ⇒ (language(x) ∧ language(y)) ∨ (city(x) ∧ city(y)) ∨ (country(x) ∧ country(y))",
            "violations": lambda graph, domain: SynTypesVerifier.rule_3_validate_same_type_as(graph),
            "description": "Validates that the entities connected by 'same_type_as' are of the same type.",
            "failure_message": "Rule 3: Mismatched types found in 'same_type_as' relationship.",
        },
        4: {
            "FOL": "∀x: language(x) ⇒ ¬country(x) ∧ ¬city(x)",
            "violations": lambda graph, domain: SynTypesVerifier.rule_4_validate_exclusive_language(graph),
            "description": "Validates that a language entity does not belong to the domain of countries or cities.",
            "failure_message": "Rule 4: A language entity cannot also belong to the domain of countries or cities.",
        },
        5: {
            "FOL": "∀x: country(x) ⇒ ¬language(x) ∧ ¬city(x)",
            "violations": lambda graph, domain: SynTypesVerifier.rule_5_validate_exclusive_country(graph),
            "description": "Validates that a country entity does not belong to the domain of languages or cities.",
            "failure_message": "Rule 5: A country entity cannot also belong to the domain of languages or cities.",
        },
        6: {
            "FOL": "∀x: city(x) ⇒ ¬language(x) ∧ ¬country(x)",
            "violations": lambda graph, domain: SynTypesVerifier.rule_6_validate_exclusive_city(graph),
            "description": "Validates that a city entity does not belong to the domain of languages or countries.",
            "failure_message": "Rule 6: A city entity cannot also belong to the domain of languages or countries.",
        },
        7: {
            "FOL": "∀x, y, z : (connected(x, y) ∧ connected(y, z)) ⇒ connected(x, z)",
            "violations": lambda graph, domain: SynTypesVerifier.rule_7_validate_connectedness(graph),
            "description": "Validates that all triples in the graph are connected.",
            "failure_message": "Rule 7: Graph is not fully connected.",
        },
        8: {
            "FOL": "∀x, y : relationship(x, y) ⇒ (spoken_in(x, y) ∨ could_be_part_of(x, y) ∨ same_type_as(x, y))",
            "violations": lambda graph, domain: SynTypesVerifier.rule_8_validate_allowed_relationships(graph),
            "description": "Validates that no invalid relationships are present in the graph.",
            "failure_message": "Rule 8: Invalid relationship type found.",
        },
        9: {
            "FOL": "∀x, y : relationship(x, y) ⇒ x ≠ y",
            "violations": lambda graph, domain: SynTypesVerifier.rule_9_validate_no_self_loops(graph),
            "description": "Validates that there are no self-loops.",
            "failure_message": "Rule 9: Self-loop detected in the graph.",
        },
        10: {
            "FOL": "count(graph) = 10",
            # Traditional FOL doesn't natively support counting quantifiers or direct cardinality comparisons
            "violations": lambda graph, domain: SynTypesVerifier.rule_10_check_triple_count(graph, expected_count=10),
            "description": "Ensures the graph contains exactly the expected number of triples.",
            "failure_message": "Rule 14: The number of triples in the graph does not match the expected count.",
        },
    }
