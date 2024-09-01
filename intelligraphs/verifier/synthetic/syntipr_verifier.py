from intelligraphs.verifier.constraint_verifier_base import ConstraintVerifier
from intelligraphs.domains.SynTIPR.entities import names, roles, years
from intelligraphs.domains.SynTIPR.relations import relations


class SynTIPRVerifier(ConstraintVerifier):
    # Subclass definition for the syn-TIPR dataset
    DOMAIN = {
        "academic": "_academic",
        "time": "_time",
        "names": names,
        "roles": roles,
        "years": years
    }

    RELATIONS = relations

    @staticmethod
    def rule_1_has_role_constraint(graph, domain):
        return {
            (s, p, o) for s, p, o in graph
            if p == "has_role" and not (s == domain['academic'] and o in domain['roles'])
        }

    @staticmethod
    def rule_2_has_name_constraint(graph, domain):
        return {
            (s, p, o) for s, p, o in graph
            if p == "has_name" and not (s == domain['academic'] and o in domain['names'])
        }

    @staticmethod
    def rule_3_has_time_constraint(graph, domain):
        return {
            (s, p, o) for s, p, o in graph
            if p == "has_time" and not (s == domain['academic'] and o == domain['time'])
        }

    @staticmethod
    def rule_4_start_year_constraint(graph, domain):
        return {
            (s, p, o) for s, p, o in graph
            if p == "start_year" and not (s == domain['time'] and o in domain['years'])
        }

    @staticmethod
    def rule_5_end_year_constraint(graph, domain):
        return {
            (s, p, o) for s, p, o in graph
            if p == "end_year" and not (s == domain['time'] and o in domain['years'])
        }

    @staticmethod
    def rule_6_end_year_after_start_year(graph, domain):
        violations = set()

        start_years = {(s, o) for s, p, o in graph if p == "start_year"}
        end_years = {(s, o) for s, p, o in graph if p == "end_year"}

        for subject, start_year in start_years:
            for subj, end_year in end_years:
                if subject == subj and isinstance(end_year, int) and isinstance(start_year, int):
                    if start_year > end_year:
                        violations.add((subject, "start_year", start_year))
                        violations.add((subject, "end_year", end_year))

        return violations

    @staticmethod
    def rule_7_no_self_role(graph, domain):
        return {
            (s, p, o) for s, p, o in graph if p == "has_role" and s == o
        }

    @staticmethod
    def rule_8_no_self_name(graph, domain):
        return {
            (s, p, o) for s, p, o in graph if p == "has_name" and s == o
        }

    @staticmethod
    def rule_9_no_self_time(graph, domain):
        return {
            (s, p, o) for s, p, o in graph if p == "has_time" and s == o
        }

    @staticmethod
    def rule_10_no_self_start_year(graph, domain):
        return {
            (s, p, o) for s, p, o in graph if p == "start_year" and s == o
        }

    @staticmethod
    def rule_11_no_self_end_year(graph, domain):
        return {
            (s, p, o) for s, p, o in graph if p == "end_year" and s == o
        }

    @staticmethod
    def rule_12_academic_not_role_time_name_year(graph, domain):
        violations = set()

        for s, p, o in graph:
            if s == domain['academic']:
                if s in domain['roles'] or s in domain['names'] or s in domain['years'] or s == domain['time']:
                    violations.add((s, p, o))
            if o == domain['academic']:
                if o in domain['roles'] or o in domain['names'] or o in domain['years'] or o == domain['time']:
                    violations.add((s, p, o))

        return violations

    @staticmethod
    def rule_14_time_not_academic_role_name_year(graph, domain):
        violations = set()

        for s, p, o in graph:
            if s == domain['time']:
                if s in domain['names'] or s in domain['years'] or s == domain['roles'] or s == domain['academic']:
                    violations.add((s, p, o))
            if o == domain['time']:
                if o in domain['names'] or o in domain['years'] or o == domain['roles'] or o == domain['academic']:
                    violations.add((s, p, o))

        return violations

    @staticmethod
    def rule_13_role_not_academic_time_name_year(graph, domain):
        violations = set()

        for s, p, o in graph:
            if s == domain['roles']:
                if s in domain['names'] or s in domain['years'] or s == domain['time'] or s == domain['academic']:
                    violations.add((s, p, o))
            if o == domain['roles']:
                if o in domain['names'] or o in domain['years'] or o == domain['time'] or o == domain['academic']:
                    violations.add((s, p, o))

        return violations

    @staticmethod
    def rule_15_year_not_academic_role_time_name(graph, domain):
        violations = set()

        for s, p, o in graph:
            if s == domain['years']:
                if s in domain['names'] or s in domain['roles'] or s == domain['academic'] or s == domain['time']:
                    violations.add((s, p, o))
            if o == domain['years']:
                if o in domain['names'] or o in domain['roles'] or o == domain['academic'] or o == domain['time']:
                    violations.add((s, p, o))

        return violations

    @staticmethod
    def rule_16_name_not_academic_role_time_year(graph, domain):
        violations = set()

        for s, p, o in graph:
            if s in domain['names']:
                if s in domain['roles'] or s in domain['years'] or s == domain['academic'] or s == domain['time']:
                    violations.add((s, p, o))

        return violations

    @staticmethod
    def rule_17_unique_end_year(graph, domain):
        violations = set()
        end_years = [(s, o) for s, p, o in graph if p == "end_year"]

        if len(end_years) > 1:
            violations.update({(s, "end_year", o) for s, o in end_years})
        elif len(end_years) == 0:
            violations.add(("<missing>", "end_year", "<missing>"))

        return violations

    @staticmethod
    def rule_18_unique_has_role(graph, domain):
        violations = set()
        has_roles = [(s, o) for s, p, o in graph if p == "has_role"]

        if len(has_roles) > 1:
            violations.update({(s, "has_role", o) for s, o in has_roles})
        elif len(has_roles) == 0:
            violations.add(("<missing>", "has_role", "<missing>"))

        return violations

    @staticmethod
    def rule_19_unique_has_name(graph, domain):
        violations = set()
        has_names = [(s, o) for s, p, o in graph if p == "has_name"]

        if len(has_names) > 1:
            violations.update({(s, "has_name", o) for s, o in has_names})
        elif len(has_names) == 0:
            violations.add(("<missing>", "has_name", "<missing>"))

        return violations

    @staticmethod
    def rule_20_unique_start_year(graph, domain):
        violations = set()
        start_years = [(s, o) for s, p, o in graph if p == "start_year"]

        if len(start_years) > 1:
            violations.update({(s, "start_year", o) for s, o in start_years})
        elif len(start_years) == 0:
            violations.add(("<missing>", "start_year", "<missing>"))

        return violations

    @staticmethod
    def rule_21_unique_has_time(graph, domain):
        violations = set()
        has_times = [(s, o) for s, p, o in graph if p == "has_time"]

        if len(has_times) > 1:
            violations.update({(s, "has_time", o) for s, o in has_times})
        elif len(has_times) == 0:
            violations.add(("<missing>", "has_time", "<missing>"))

        return violations

    @staticmethod
    def rule_22_triple_count_constraint(graph, expected_count):
        return graph if len(graph) != expected_count else set()

    # Violations should return a set of tuples of the form (subject, predicate, object) that violate the rule (if any)
    RULES = {
        1: {
            "FOL": "∀x, y : has_role(x, y) ⇒ academic(x) ∧ role(y)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_1_has_role_constraint(graph, domain),
            "description": "If an entity has a role, then the subject must be an academic, "
                           "and the object must be a valid role.",
            "failure_message": "Rule 1: If an entity has a role, it cannot be anything other than an academic "
                               "or have an unrecognized role.",
        },
        2: {
            "FOL": "∀x, y : has_name(x, y) ⇒ academic(x) ∧ name(y)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_2_has_name_constraint(graph, domain),
            "description": "If an entity has a name, then the entity must be an academic, "
                           "and the name must be recognized.",
            "failure_message": "Rule 2: If an entity has a name, it cannot be anything other than an academic "
                               "or have an unrecognized name.",
        },
        3: {
            "FOL": "∀x, y : has_time(x, y) ⇒ academic(x) ∧ time(y)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_3_has_time_constraint(graph, domain),
            "description": "If an entity has a time, then the entity must be an academic, "
                           "and the time must be recognized.",
            "failure_message": "Rule 3: If an entity has a time, it cannot be anything other than an academic "
                               "or have an unrecognized time.",
        },
        4: {
            "FOL": "∀x, y : start_year(x, y) ⇒ time(x) ∧ year(y)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_4_start_year_constraint(graph, domain),
            "description": "If there's a start year, the entity representing it must be a time and "
                           "the year must be recognized.",
            "failure_message": "Rule 4: If an entity denotes a start year, it cannot be anything other "
                               "than a time or be linked to an unrecognized year.",
        },
        5: {
            "FOL": "∀x, y : end_year(x, y) ⇒ time(x) ∧ year(y)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_5_end_year_constraint(graph, domain),
            "description": "If there's an end year, the entity representing it must be a time and "
                           "the year must be recognized.",
            "failure_message": "Rule 5: If an entity denotes an end year, it cannot be anything other "
                               "than a time or be linked to an unrecognized year.",
        },
        6: {
            "FOL": "∀x, y, z : end_year(x, y) ∧ start_year(x, z) ⇒ after(y, z)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_6_end_year_after_start_year(graph, domain),
            "description": "An academic's tenure end year must come after its start year.",
            "failure_message": "Rule 6: An academic's tenure end year cannot be before its start year.",
        },
        7: {
            "FOL": "∀x : ¬has_role(x, x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_7_no_self_role(graph, domain),
            "description": "An entity should not have itself as a role.",
            "failure_message": "Rule 7: An entity cannot have itself as a role.",
        },
        8: {
            "FOL": "∀x : ¬has_name(x, x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_8_no_self_name(graph, domain),
            "description": "An entity should not have itself as a name.",
            "failure_message": "Rule 8: An entity cannot have itself as a name.",
        },
        9: {
            "FOL": "∀x : ¬has_time(x, x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_9_no_self_time(graph, domain),
            "description": "An entity should not have itself as a time.",
            "failure_message": "Rule 9: An entity cannot have itself as a time.",
        },
        10: {
            "FOL": "∀x : ¬start_year(x, x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_10_no_self_start_year(graph, domain),
            "description": "An entity shouldn't be its own start year.",
            "failure_message": "Rule 10: An entity cannot be its own start year.",
        },
        11: {
            "FOL": "∀x : ¬end_year(x, x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_11_no_self_end_year(graph, domain),
            "description": "An entity shouldn't be its own end year.",
            "failure_message": "Rule 11: An entity cannot be its own end year.",
        },
        12: {
            "FOL": "∀x : academic(x) ⇒ ¬role(x) ∧ ¬time(x) ∧ ¬name(x) ∧ ¬year(x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_12_academic_not_role_time_name_year(graph, domain),
            "description": "An entity identified as an academic cannot be a role, time, name, or year.",
            "failure_message": "Rule 12: An academic cannot also be a role, time, name, or year.",
        },
        13: {
            "FOL": "∀x : role(x) ⇒ ¬academic(x) ∧ ¬time(x) ∧ ¬name(x) ∧ ¬year(x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_13_role_not_academic_time_name_year(graph, domain),
            "description": "An entity identified as a role cannot be an academic, time, name, or year.",
            "failure_message": "Rule 13: A role cannot also be an academic, time, name, or year.",
        },
        14: {
            "FOL": "∀x : time(x) ⇒ ¬academic(x) ∧ ¬role(x) ∧ ¬name(x) ∧ ¬year(x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_14_time_not_academic_role_name_year(graph, domain),
            "description": "An entity identified as a time cannot be an academic, role, name, or year.",
            "failure_message": "Rule 14: A time cannot also be an academic, role, name, or year.",
        },
        15: {
            "FOL": "∀x : year(x) ⇒ ¬academic(x) ∧ ¬role(x) ∧ ¬name(x) ∧ ¬time(x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_15_year_not_academic_role_time_name(graph, domain),
            "description": "An entity identified as a year cannot be an academic, role, time, or name.",
            "failure_message": "Rule 15: A year cannot also be an academic, role, time, or name.",
        },
        16: {
            "FOL": "∀x : name(x) ⇒ ¬academic(x) ∧ ¬role(x) ∧ ¬year(x) ∧ ¬time(x)",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_16_name_not_academic_role_time_year(graph, domain),
            "description": "An entity identified as a name cannot be an academic, role, time, or year.",
            "failure_message": "Rule 16: A name cannot also be an academic, role, time, or year.",
        },
        17: {
            # Rule below are based on Uniqueness quantification (https://en.wikipedia.org/wiki/Uniqueness_quantification)
            "FOL": "∃x,a ( end_year(x, a) ∧ ¬∃y,b ( end_year(y, b) ∧ y ≠ x ∧ b ≠ a))",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_17_unique_end_year(graph, domain),
            "description": "The predicate 'end_year' should occur only once in the graph.",
            "failure_message": "Rule 17: The predicate 'end_year' did not occur or "
                               "occurred more than once in the graph.",
        },
        18: {
            # Rule below are based on Uniqueness quantification (https://en.wikipedia.org/wiki/Uniqueness_quantification)
            "FOL": "∃x( has_role(x) ∧ ¬∃y( has_role(y) ∧ y ≠ x))",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_18_unique_has_role(graph, domain),
            "description": "The predicate 'has_role' should occur only once in the graph.",
            "failure_message": "Rule 18: The predicate 'has_role' did not occur or "
                               "occurred more than once in the graph.",
        },
        19: {
            # Rule below are based on Uniqueness quantification (https://en.wikipedia.org/wiki/Uniqueness_quantification)
            "FOL": "∃x( has_name(x) ∧ ¬∃y( has_name(y) ∧ y ≠ x))",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_19_unique_has_name(graph, domain),
            "description": "The predicate 'has_name' should occur only once in the graph.",
            "failure_message": "Rule 19: The predicate 'has_name' did not occur or "
                               "occurred more than once in the graph.",
        },
        20: {
            # Rule below are based on Uniqueness quantification (https://en.wikipedia.org/wiki/Uniqueness_quantification)
            "FOL": "∃x( start_year(x) ∧ ¬∃y( start_year(y) ∧ y ≠ x))",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_20_unique_start_year(graph, domain),
            "description": "The predicate 'start_year' should occur only once in the graph.",
            "failure_message": "Rule 20: The predicate 'start_year' did not occur or "
                               "occurred more than once in the graph.",
        },
        21: {
            # Rule below are based on Uniqueness quantification (https://en.wikipedia.org/wiki/Uniqueness_quantification)
            "FOL": "∃x( has_time(x) ∧ ¬∃y( has_time(y) ∧ y ≠ x))",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_21_unique_has_time(graph, domain),
            "description": "The predicate 'has_time' should occur only once in the graph.",
            "failure_message": "Rule 21: The predicate 'has_time' did not occur or "
                               "occurred more than once in the graph.",
        },
        22: {
            "FOL": "count(graph) = 5",
            "violations": lambda graph, domain: SynTIPRVerifier.rule_22_triple_count_constraint(graph, 5),
            "description": "Ensures the graph contains exactly the expected number of triples.",
            "failure_message": "Rule 22: The number of triples in the graph does not match the expected count.",
        },
    }
