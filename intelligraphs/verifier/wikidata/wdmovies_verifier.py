from ..constraint_verifier_base import ConstraintVerifier
from ...domains.WDMovies.entities import actors, directors, genres
from ...domains.WDMovies.relations import relations
from typing import List, Tuple, Set


class WDMoviesVerifier(ConstraintVerifier):
    # Subclass definition for WDMovies
    DOMAIN = {
        "existential_node": "_movie",
        "actor": actors,
        "director": directors,
        "genre": genres,
    }
    DOMAIN["person"] = set(list(DOMAIN["director"]) + list(DOMAIN["actor"]))

    RELATIONS = relations

    @staticmethod
    def rule_1_validate_connected_relationships(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that the 'connected' relationship holds only if there is a valid 'has_director',
        'has_actor', or 'has_genre' relationship.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if predicate not in {'has_director', 'has_actor', 'has_genre'}:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_2_validate_has_director(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that there is at least one 'has_director' relationship involving the existential node.
        """
        violations = set()
        if not any(predicate == "has_director" and subject == WDMoviesVerifier.DOMAIN['existential_node'] for
                   subject, predicate, obj in graph):
            violations.add((WDMoviesVerifier.DOMAIN['existential_node'], "has_director", "<missing_director>"))
        return violations

    @staticmethod
    def rule_3_validate_has_actor(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that there is at least one 'has_actor' relationship involving the existential node.
        """
        violations = set()
        if not any(predicate == "has_actor" and subject == WDMoviesVerifier.DOMAIN['existential_node'] for
                   subject, predicate, obj in graph):
            violations.add((WDMoviesVerifier.DOMAIN['existential_node'], "has_actor", "<missing_actor>"))
        return violations

    @staticmethod
    def rule_4_validate_connected_to_existential_node(graph: List[Tuple[str, str, str]]) -> Set[
        Tuple[str, str, str]]:
        """
        Validates that every node (except the existential node itself) is connected to the existential node.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if obj != WDMoviesVerifier.DOMAIN["existential_node"] and subject != WDMoviesVerifier.DOMAIN[
                "existential_node"]:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_5_validate_no_connections_without_existential_node(graph: List[Tuple[str, str, str]]) -> Set[
        Tuple[str, str, str]]:
        """
        Validates that no two nodes are connected unless one of them is the existential node.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if subject != WDMoviesVerifier.DOMAIN["existential_node"] and obj != WDMoviesVerifier.DOMAIN[
                "existential_node"]:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_6_validate_no_connections_to_existential_node(graph: List[Tuple[str, str, str]]) -> Set[
        Tuple[str, str, str]]:
        """
        Validates that no node is connected to the existential node.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if obj == WDMoviesVerifier.DOMAIN["existential_node"]:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_7_validate_person_relationships(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that if a node has a 'has_director' or 'has_actor' relationship, the object must be a person.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if predicate in {'has_director', 'has_actor'} and obj not in WDMoviesVerifier.DOMAIN['person']:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_8_validate_person_genre_exclusivity(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that no entity can be classified as both a person and a genre.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if predicate in {'has_director', 'has_actor'} and obj in WDMoviesVerifier.DOMAIN['genre']:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_9_validate_genre_relationships(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that if a node has a 'has_genre' relationship, the object must be a genre.
        """
        violations = set()
        for subject, predicate, obj in graph:
            if predicate == 'has_genre' and obj not in WDMoviesVerifier.DOMAIN['genre']:
                violations.add((subject, predicate, obj))
        return violations

    # Violations should return a set of tuples of the form (subject, predicate, object) that violate the rule (if any)
    RULES = {
        1: {
            "FOL": "∀x, y: connected(x, y) ⇔ has_director(x, y) ∨ has_actor(x, y) ∨ has_genre(x, y)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_1_validate_connected_relationships(graph),
            "description": "x is connected to y if and only if x has y as a director, actor, or genre.",
            "failure_message": "Rule 1: x is connected to y, but x does not have y as a director, actor, or genre.",
        },
        2: {
            "FOL": "∃x: has_director(x, existential_node)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_2_validate_has_director(graph),
            "description": "There exists an x that has the existential_node as a director.",
            "failure_message": "Rule 2: There is no x that has the existential_node as a director.",
        },
        3: {
            "FOL": "∃x: has_actor(x, existential_node)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_3_validate_has_actor(graph),
            "description": "There exists an x that has the existential_node as an actor.",
            "failure_message": "Rule 3: There is no x that has the existential_node as an actor.",
        },
        4: {
            "FOL": "∀x: x ≠ existential_node ⇒ connected(existential_node, x)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_4_validate_connected_to_existential_node(graph),
            "description": "For every x not equal to the existential_node, the existential_node is connected to x.",
            "failure_message": "Rule 5: x is not equal to the existential_node, but the existential_node is not connected to x.",
        },
        5: {
            "FOL": "∀x, y: x ≠ existential_node ∧ y ≠ existential_node ⇒ ¬connected(x, y)",
            "violations": lambda graph,
                                 domain: WDMoviesVerifier.rule_5_validate_no_connections_without_existential_node(
                graph),
            "description": "For every x and y not equal to the existential_node, x is not connected to y.",
            "failure_message": "Rule 6: x and y are not equal to the existential_node, but x is connected to y.",
        },
        6: {
            "FOL": "∀x: ¬connected(x, existential_node)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_6_validate_no_connections_to_existential_node(
                graph),
            "description": "No x is connected to the existential_node.",
            "failure_message": "Rule 7: x is connected to the existential_node.",
        },
        7: {
            "FOL": "∀x, y: has_director(x, y) ∨ has_actor(x, y) ⇒ person(y)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_7_validate_person_relationships(graph),
            "description": "If x has y as a director or actor, then y is a person.",
            "failure_message": "Rule 8: x has y as a director or actor, but y is not a person.",
        },
        8: {
            "FOL": "∀x: ¬person(x) ∨ ¬genre(x)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_8_validate_person_genre_exclusivity(graph),
            "description": "No x can be both a person and a genre.",
            "failure_message": "Rule 9: x is both a person and a genre.",
        },
        9: {
            "FOL": "∀x, y: has_genre(x, y) ⇒ genre(y)",
            "violations": lambda graph, domain: WDMoviesVerifier.rule_9_validate_genre_relationships(graph),
            "description": "If x has y as a genre, then y is a genre.",
            "failure_message": "Rule 10: x has y as a genre, but y is not a genre.",
        }
    }

