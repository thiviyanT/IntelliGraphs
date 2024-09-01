from intelligraphs.verifier.constraint_verifier_base import ConstraintVerifier
from intelligraphs.domains.WDArticles.entities import article_node, authors, names, articles, ordinals, subjects, iri
from intelligraphs.domains.WDArticles import relations
from typing import List, Tuple, Set


class WDArticlesVerifier(ConstraintVerifier):
    # Subclass definition for WDArticles
    DOMAIN = {
        "article": article_node.union(articles),
        'authors': authors,
        'names': names,
        'ordinals': ordinals,
        'subjects': subjects,
        'iri': iri,
    }

    RELATIONS = relations

    @staticmethod
    def rule_1_validate_has_author(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        for subject, predicate, obj in graph:
            if predicate == "has_author" and subject in WDArticlesVerifier.DOMAIN["article"]:
                return set()  # No violation
        return {("_article", "has_author", "<missing_author>")}

    @staticmethod
    def rule_2_validate_connected(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        valid_relations = {"has_author", "has_name", "has_order", "cites", "has_subject", "subclass_of"}
        violations = set()
        for subject, predicate, obj in graph:
            if predicate not in valid_relations:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_3_validate_no_bidirectional_edges_except_cites(graph: List[Tuple[str, str, str]]) -> Set[
        Tuple[str, str, str]]:
        violations = set()
        edge_set = {(s, o) for s, p, o in graph if p != "cites"}

        for s, p, o in graph:
            if p != "cites" and (o, s) in edge_set:
                violations.add((s, p, o))
                violations.add((o, p, s))

        return violations

    @staticmethod
    def rule_4_validate_no_self_loops(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        return {(s, p, o) for s, p, o in graph if s == o}

    @staticmethod
    def rule_5_validate_author_relationship(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        for subject, predicate, obj in graph:
            if predicate == "has_author" and subject not in WDArticlesVerifier.DOMAIN["article"]:
                violations.add((subject, predicate, obj))
        return violations

    @staticmethod
    def rule_6_validate_authorpos_complete_relationship(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        authorpos_nodes = set()

        # Collect all author_pos nodes
        for subject, predicate, _ in graph:
            if subject in WDArticlesVerifier.DOMAIN['authors']:
                authorpos_nodes.add(subject)

        # Check that each author_pos node has both 'has_order' and 'has_name' relationships
        for author_pos in authorpos_nodes:
            has_order = any(s == author_pos and p == "has_order" for s, p, o in graph)
            has_name = any(s == author_pos and p == "has_name" for s, p, o in graph)
            if not (has_order and has_name):
                violations.add((author_pos, "incomplete_authorpos", author_pos))

        return violations

    @staticmethod
    def rule_7_validate_has_order_relationship(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()

        for subject, predicate, obj in graph:
            if predicate == "has_order":
                if subject not in WDArticlesVerifier.DOMAIN['authors']:
                    violations.add((subject, predicate, obj))
                if obj not in WDArticlesVerifier.DOMAIN['ordinals']:
                    violations.add((subject, predicate, obj))

        return violations

    @staticmethod
    def rule_8_validate_has_name_relationship(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()

        for subject, predicate, obj in graph:
            if predicate == "has_name":
                if subject not in WDArticlesVerifier.DOMAIN['authors']:
                    violations.add((subject, predicate, obj))
                if obj not in WDArticlesVerifier.DOMAIN['names']:
                    violations.add((subject, predicate, obj))

        return violations

    @staticmethod
    def rule_9_validate_unique_order_per_authorpos(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Validates that an author position node cannot have multiple 'has_order' relationships with different ordinals.
        """
        violations = set()
        authorpos_to_order = {}

        for subject, predicate, obj in graph:
            if predicate == "has_order":
                if subject in authorpos_to_order:
                    if authorpos_to_order[subject] != obj:
                        # Violation found: the same author position has multiple different orders
                        violations.add((subject, "conflicting_has_order", f"{authorpos_to_order[subject]} vs {obj}"))
                else:
                    authorpos_to_order[subject] = obj

        return violations

    @staticmethod
    def rule_10_validate_unique_name_per_authorpos(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        authorpos_to_names = {}

        for subject, predicate, obj in graph:
            if predicate == "has_name":
                if subject in authorpos_to_names:
                    if authorpos_to_names[subject] != obj:
                        # Violation: the same author_pos has multiple different names
                        violations.add((subject, "conflicting_has_name", f"{authorpos_to_names[subject]} vs {obj}"))
                else:
                    authorpos_to_names[subject] = obj

        return violations

    @staticmethod
    def rule_11_validate_subject_type_exclusivity(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        subjects = set()

        for subject, predicate, obj in graph:
            if subject in WDArticlesVerifier.DOMAIN['subjects']:
                subjects.add(subject)
            if obj in WDArticlesVerifier.DOMAIN['subjects']:
                subjects.add(obj)

        for subject_entity in subjects:
            if (subject_entity in WDArticlesVerifier.DOMAIN['ordinals'] or
                    subject_entity in WDArticlesVerifier.DOMAIN['authors']):
                violations.add((subject_entity, "conflicting_type", "subject"))

        return violations

    @staticmethod
    def rule_12_validate_iri_type_exclusivity(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        iris = set()

        for subject, predicate, obj in graph:
            if obj in WDArticlesVerifier.DOMAIN['iri']:
                iris.add(obj)

            if subject in WDArticlesVerifier.DOMAIN['iri']:
                iris.add(subject)

        for iri in iris:
            if (iri in WDArticlesVerifier.DOMAIN['ordinals'] or
                    iri in WDArticlesVerifier.DOMAIN['authors']):
                violations.add((iri, "conflicting_type", "iri"))

        return violations

    @staticmethod
    def rule_13_validate_name_type_exclusivity(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        names = set()

        for subject, predicate, obj in graph:
            if subject in WDArticlesVerifier.DOMAIN['names']:
                names.add(subject)

            if obj in WDArticlesVerifier.DOMAIN['names']:
                names.add(obj)

        for name in names:
            if (name in WDArticlesVerifier.DOMAIN['ordinals'] or
                    name in WDArticlesVerifier.DOMAIN['authors']):
                violations.add((name, "conflicting_type", "name1"))

        return violations

    @staticmethod
    def rule_14_validate_ordinal_type_exclusivity(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        ordinals = set()

        # Gather all entities classified as 'ordinal'
        for subject, predicate, obj in graph:
            if subject in WDArticlesVerifier.DOMAIN['ordinals']:
                ordinals.add(subject)

            if obj in WDArticlesVerifier.DOMAIN['ordinals']:
                ordinals.add(obj)

        # Check if any 'ordinal' entity is also classified as 'subject', 'iri', 'name', or 'author'
        for ordinal in ordinals:
            if (ordinal in WDArticlesVerifier.DOMAIN['subjects'] or
                    ordinal in WDArticlesVerifier.DOMAIN['iri'] or
                    ordinal in WDArticlesVerifier.DOMAIN['names'] or
                    ordinal in WDArticlesVerifier.DOMAIN['authors']):
                violations.add((ordinal, "conflicting_type", "ordinal"))

        return violations

    @staticmethod
    def rule_15_validate_author_type_exclusivity(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()
        authors = set()

        # Gather all entities classified as 'author'
        for subject, predicate, obj in graph:
            if subject in WDArticlesVerifier.DOMAIN['authors']:
                authors.add(subject)
            if obj in WDArticlesVerifier.DOMAIN['authors']:
                authors.add(obj)

        # Check if any 'author' entity is also classified as 'subject', 'iri', 'name', or 'ordinal'
        for author in authors:
            if (author in WDArticlesVerifier.DOMAIN['subjects'] or
                    author in WDArticlesVerifier.DOMAIN['iri'] or
                    author in WDArticlesVerifier.DOMAIN['names'] or
                    author in WDArticlesVerifier.DOMAIN['ordinals']):
                violations.add((author, "conflicting_type", "author"))

        return violations

    @staticmethod
    def rule_17_subclass_trans(graph: List[Tuple[str, str, str]], x: str, z: str) -> bool:
        """Recursive function to check if there is a transitive subclass_of path from x to z."""
        if x == z:
            return True
        for subject, predicate, obj in graph:
            if predicate == "subclass_of" and subject == x:
                if obj == z or WDArticlesVerifier.rule_17_subclass_trans(graph, obj, z):
                    return True
        return False

    @staticmethod
    def rule_17_validate_subclass_of_relationship(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Ensures that every subclass_of relationship has a corresponding valid transitive path
        and that both the subject and object are either an iri or a subject.
        """
        violations = set()

        # Extract relevant relationships
        subclass_of_relations = [(s, o) for s, p, o in graph if p == "subclass_of"]

        # Create a set of valid entities (iri or subject)
        valid_entities = WDArticlesVerifier.DOMAIN['iri'].union(WDArticlesVerifier.DOMAIN['subjects'])

        # Check each subclass_of relationship
        for x, y in subclass_of_relations:
            # Ensure that both x and y are valid entities (iri or subject)
            if x not in valid_entities or y not in valid_entities:
                violations.add((x, "invalid_entity_type_for_subclass_of", y))

            # Check if there is a valid transitive path from x to y
            if not WDArticlesVerifier.rule_17_subclass_trans(graph, x, y):
                violations.add((x, "missing_transitive_closure", y))

        return violations

    @staticmethod
    def rule_18_validate_subclass_of_linked_to_subject(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """
        Ensures that every subclass_of relationship is eventually linked back to a subject
        through a transitive closure and a has_subject relationship.
        """
        violations = set()

        # Build the subclass map and identify all subject nodes
        subclass_map = {}
        subject_nodes = set()

        for subject, predicate, obj in graph:
            if predicate == "subclass_of":
                if obj not in subclass_map:
                    subclass_map[obj] = []
                subclass_map[obj].append(subject)
            elif predicate == "has_subject":
                subject_nodes.add(obj)

        # Function to check if a node eventually links to a subject node
        def is_linked_to_subject(node, visited):
            if node in subject_nodes:
                return True
            if node in visited:
                return False
            if node not in subclass_map:
                return False

            visited.add(node)
            for parent in subclass_map[node]:
                if is_linked_to_subject(parent, visited):
                    return True
            visited.remove(node)
            return False

        # Check each subclass_of relationship
        for subject, predicate, obj in graph:
            if predicate == "subclass_of":
                if not is_linked_to_subject(subject, set()):
                    violations.add((subject, "subclass_of_not_linked_to_subject", obj))
        return violations

    @staticmethod
    def rule_19_validate_cites_relationship(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()

        for subject, predicate, obj in graph:
            if predicate == "cites":
                if subject not in WDArticlesVerifier.DOMAIN['article']:
                    violations.add((subject, predicate, obj))
                if obj not in WDArticlesVerifier.DOMAIN['iri']:
                    violations.add((subject, predicate, obj))

        return violations

    @staticmethod
    def rule_20_validate_has_subject_relationship(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        violations = set()

        for subject, predicate, obj in graph:
            if predicate == "has_subject":
                if subject not in WDArticlesVerifier.DOMAIN['article']:
                    violations.add((subject, predicate, obj))
                if obj not in WDArticlesVerifier.DOMAIN['subjects'] and obj not in WDArticlesVerifier.DOMAIN['iri']:
                    violations.add((subject, predicate, obj))

        return violations

    @staticmethod
    def rule_21_validate_consecutive_ordinals(graph: List[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        ordinal_values = []
        violations = set()

        for subject, predicate, obj in graph:
            if predicate == "has_order" and obj in WDArticlesVerifier.DOMAIN["ordinals"]:
                # Assuming ordinals are named in the format 'ordinal_XXX' where XXX is a zero-padded number
                # Safety check to ensure correct format 'ordinal_XXX'
                if obj.startswith("ordinal_") and obj.split('_')[-1].isdigit():
                    try:
                        ordinal_values.append(int(obj.split('_')[-1]))
                    except ValueError:
                        violations.add((subject, "invalid_ordinal_format", obj))
                else:
                    violations.add((subject, "invalid_ordinal_format", obj))

        ordinal_values.sort()

        # Ordinal sequence should start from 1
        expected_start = 1
        if ordinal_values and ordinal_values[0] != expected_start:
            violations.add((f"ordinal_{expected_start:03}", "missing_starting_ordinal",
                            f"actual_starting_ordinal_{ordinal_values[0]:03}"))

        # Check for consecutive sequence
        for i in range(1, len(ordinal_values)):
            if ordinal_values[i] != ordinal_values[i - 1] + 1:
                expected_ordinal = ordinal_values[i - 1] + 1
                actual_ordinal = ordinal_values[i]
                violations.add((f"ordinal_{expected_ordinal:03}", "missing_consecutive_ordinal",
                                f"actual_ordinal_{actual_ordinal:03}"))

        return violations

    RULES = {
        1: {
            "FOL": "∃x: has_author(article_node, x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_1_validate_has_author(graph),
            "description": "Ensures that there is at least one 'has_author' relationship for an article node.",
            "failure_message": "Rule 1: Invalid 'has_author' relationship found.",
        },
        2: {
            "FOL": "∀x,y: connected(x,y) ⇔ has_author(x,y) ∨ has_name(x,y) ∨ has_order(x,y) ∨ cites(x,y) ∨ has_subject(x,y) ∨ subclass_of(x,y)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_2_validate_connected(graph),
            "description": "Ensures that all connected nodes are linked by valid relationships.",
            "failure_message": "Rule 2: Invalid relationship found between nodes.",
        },
        3: {
            "FOL": "∀x,y: connected(x,y) ⇒ ¬connected(y,x) ∨ cites(y, x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_3_validate_no_bidirectional_edges_except_cites(graph),
            "description": "Ensures that connections are not bidirectional unless 'cites' is involved.",
            "failure_message": "Rule 3: Bidirectional connection found where it shouldn't exist.",
        },
        4: {
            "FOL": "∀x: ¬connected(x, x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_4_validate_no_self_loops(graph),
            "description": "Ensures that no node is connected to itself (no self-loops).",
            "failure_message": "Rule 4: Self-loop detected in the graph.",
        },
        5: {
            "FOL": "∀x, y: has_author(x, y) ⇒ x = article_node ∧ (article(article_node) ∨ iri(article_node))",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_5_validate_author_relationship(graph),
            "description": "Ensures that 'has_author' relationships only involve article nodes as subjects and the article_node is either an article or an IRI.",
            "failure_message": "Rule 5: Invalid 'has_author' relationship: subject must be an article node.",
        },
        6: {
            "FOL": "∀x: has_author(article_node, x) ⇒ author(x) ∀x: author(x) ⇔ ∃y: has_order(x, y) ∧ ∃y: has_name(x, y)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_6_validate_authorpos_complete_relationship(graph),
            "description": "Ensures that every author position is linked to both an ordinal (via 'has_order') and a name (via 'has_name').",
            "failure_message": "Rule 6: Author position node is missing either 'has_order' or 'has_name' relationship.",
        },
        7: {
            "FOL": "∀x,y: has_order(x,y) ⇒ author(x) ∧ ordinal(y)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_7_validate_has_order_relationship(graph),
            "description": "Ensures that 'has_order' relationships correctly link an author position and an ordinal.",
            "failure_message": "Rule 8: Invalid 'has_order' relationship found.",
        },
        8: {
            "FOL": "∀x, y: has_name(x, y) ⇒ author(x) ∧ (name(y) ∨ iri(y))",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_8_validate_has_name_relationship(graph),
            "description": "Validates that 'has_name' relationships link an author position and a name or IRI.",
            "failure_message": "Rule 9: Invalid 'has_name' relationship found.",
        },
        9: {
            "FOL": "∀x, y, z: has_order(x, y) ∧ has_order(x, z) ⇒ y = z",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_9_validate_unique_order_per_authorpos(graph),
            "description": "Ensures that an author position cannot have multiple orders.",
            "failure_message": "Rule 10: Conflicting 'has_order' relationships found for the same author position.",
        },
        10: {
            "FOL": "∀x, y, z: has_name(x, y) ∧ has_name(x, z) ⇒ y = z",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_10_validate_unique_name_per_authorpos(graph),
            "description": "Ensures that each 'author' node has a unique 'has_name' relationship.",
            "failure_message": "Rule 11: Conflicting 'has_name' relationships found for the same author position.",
        },
        11: {
            "FOL": "∀x: subject(x) ⇒ ¬ordinal(x) ∧ ¬author(x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_11_validate_subject_type_exclusivity(graph),
            "description": "Ensures that an entity classified as 'subject' is not classified as another type.",
            "failure_message": "Rule 13: Conflicting types found for an entity classified as 'subject'.",
        },
        12: {
            "FOL": "∀x: iri(x) ⇒ ¬ordinal(x) ∧ ¬author(x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_12_validate_iri_type_exclusivity(graph),
            "description": "Ensures that an entity classified as 'iri' is not classified as another type.",
            "failure_message": "Rule 14: Conflicting types found for an entity classified as 'iri'.",
        },
        13: {
            "FOL": "∀x: name(x) ⇒ ¬ordinal(x) ∧ ¬author(x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_13_validate_name_type_exclusivity(graph),
            "description": "Ensures that an entity classified as 'name' is not classified as another type.",
            "failure_message": "Rule 15: Conflicting types found for an entity classified as 'name'.",
        },
        14: {
            "FOL": "∀x: ordinal(x) ⇒ ¬subject(x) ∧ ¬iri(x) ∧ ¬name(x) ∧ ¬author(x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_14_validate_ordinal_type_exclusivity(graph),
            "description": "Ensures that an entity classified as 'ordinal' is not classified as another type.",
            "failure_message": "Rule 16: Conflicting types found for an entity classified as 'ordinal'.",
        },
        15: {
            "FOL": "∀x: author(x) ⇒ ¬subject(x) ∧ ¬iri(x) ∧ ¬name(x) ∧ ¬ordinal(x)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_15_validate_author_type_exclusivity(graph),
            "description": "Ensures that an entity classified as 'author' is not classified as another type.",
            "failure_message": "Rule 17: Conflicting types found for an entity classified as 'author'.",
        },
        16: {
            # subclass_trans is a transitive closure of the subclass_of relationship.
            # this is a definition of subclass_trans, so it doesn't need to be checked
            # because transitive subclass is not explicitly stated in the graph.
            "FOL": "∀x,y,z: subclass_trans(x, y) ∧ subclass_trans(y, z) ⇒ subclass_trans(x, z)",
            "violations": lambda graph, domain: set(),  # No check
            "description": "Ensures all transitive subclass relationships are explicitly stated.",
            "failure_message": "Rule 18: Transitive subclass not explicitly stated in the graph.",
        },
        17: {
            "FOL": "∀x,y: subclass_of(x,y) ⇒ subclass_trans(x,y) ∧ (iri(x) ∨ subject(x)) ∧ (iri(y) ∨ subject(y))",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_17_validate_subclass_of_relationship(graph),
            "description": "Validates `subclass_of` relationships and their types.",
            "failure_message": "Rule 19: Invalid `subclass_of` relationship or types.",
        },
        18: {
            "FOL": "∀x,y: subclass_of(x,y) ⇒ ∃z: subclass_trans(x,z) ∧ has_subject(article_node, z)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_18_validate_subclass_of_linked_to_subject(graph),
            "description": "Ensures `subclass_of` links back to an article node.",
            "failure_message": "Rule 20: `subclass_of` does not link back to an article node.",
        },
        19: {
            "FOL": "∀x,y: cites(x, y) ⇒ article(x) ∧ iri(y)",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_19_validate_cites_relationship(graph),
            "description": "Ensures that 'cites' relationships are correctly formed with IRI targets.",
            "failure_message": "Rule 21: Invalid 'cites' relationship found.",
        },
        20: {
            "FOL": "∀x,y: has_subject(x, y) ⇒ article(x) ∧ (subject(y) ∨ iri(y))",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_20_validate_has_subject_relationship(graph),
            "description": "Ensures that 'has_subject' relationships are correctly formed with subject or IRI targets.",
            "failure_message": "Rule 22: Invalid 'has_subject' relationship found.",
        },
        21: {
            "FOL": "∀x: author(x) ⇒ ordinals_consecutive(has_order(x, y))",
            "violations": lambda graph, domain: WDArticlesVerifier.rule_21_validate_consecutive_ordinals(graph),
            "description": "Ensures that the ordinals associated with author positions are consecutive.",
            "failure_message": "Rule 23: Non-consecutive ordinals found for author positions.",
        }
    }
