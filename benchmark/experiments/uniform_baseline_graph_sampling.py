from typing import Callable, Dict, Optional
from intelligraphs.data_loaders import load_data_as_list
from tqdm import trange
import torch, random
from intelligraphs.verifier.synthetic import (
    SynPathsVerifier,
    SynTypesVerifier,
    SynTIPRVerifier
)
from intelligraphs.verifier.wikidata import (
    WDArticlesVerifier,
    WDMoviesVerifier
)
from intelligraphs.evaluators import SemanticEvaluator
from intelligraphs.evaluators import post_process_data


def run_uniform_sampling_evaluation(config):
    """ Train baseline models on a dataset """

    (train, val, test,
     (entity_to_index, index_to_entity),
     (relation_to_index, index_to_relation),
     (min_edges, max_edges), (min_nodes, max_nodes)) = load_data_as_list(config["dataset"])

    if config["final"]:
        train, test = train + val, test
    else:
        train, test = train, val

    print(len(index_to_entity), 'entities')
    print(len(index_to_relation), 'relations')
    print(len(train), 'training graphs')
    print(len(test), 'test graphs')
    print(len(train) + len(test), 'total graphs')

    num_entities = len(index_to_entity)
    num_relations = len(index_to_relation)

    verifier_map = {
        "syn-paths": SynPathsVerifier(),
        "syn-tipr": SynTIPRVerifier(),
        "syn-types": SynTypesVerifier(),
        "wd-movies": WDMoviesVerifier(),
        "wd-articles": WDArticlesVerifier()
    }

    verifier_instance = verifier_map.get(config["dataset"], None)
    if verifier_instance is None:
        raise NotImplementedError('Function for semantic checking was not found ')
    rule_check_function = verifier_instance.check_rules_for_graph

    # Sampling under two settings:
    # 1) With the sampled entities
    sample_entities_and_structure(
        test=test,
        training_data=train,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        num_entities=num_entities,
        num_relations=num_relations,
        rule_check_function=rule_check_function)
    # 2) Sample structure only (SS)
    sample_only_structure(
        test=test,
        training_data=test,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        num_relations=num_relations,
        rule_check_function=rule_check_function)


def sample_entities_and_structure(
    test: list = None,
    training_data: list = None,
    index_to_entity: Dict[int, str] = None,
    index_to_relation: Dict[int, str] = None,
    num_entities: Optional[int] = None,
    num_relations: Optional[int] = None,
    rule_check_function: Callable = None
) -> Dict:
    """ Sample entities from entity frequency and structure from the model. """

    assert num_entities is None or isinstance(num_entities, int), "Expected 'num_entities' to be None or an int"
    assert num_relations is None or isinstance(num_relations, int), "Expected 'num_relations' to be None or an int"
    assert rule_check_function is None or callable(
        rule_check_function), "Expected 'rule_check_function' to be None or a callable"

    predicted_graph_structures = list()

    # Iterate through each graph individually
    for _ in trange(len(test)):

        # Sampling entities based on random sampling
        sampling_probabilities = torch.tensor([random.random() for _ in range(num_entities)])
        entity_sampling_mask = sampling_probabilities.bernoulli()
        sampled_entities = [e for e, p in index_to_entity.items() if entity_sampling_mask[e] == 1.0]

        # Map global indices to local indices for entities
        local_entity_map = {e: sampled_entities.index(e) for e in sampled_entities}
        local_num_entities = len(sampled_entities)

        subject_indices, predicate_indices, object_indices = [], [], []
        all_triplet_combinations = []  # All possible combinations of subject, predicate, and object

        # Loop over all possible triples
        for s in sampled_entities:
            subject_label = index_to_entity[s]

            for o in sampled_entities:
                object_label = index_to_entity[o]

                for p in range(len(index_to_relation)):
                    predicate_label = index_to_relation[p]

                    subject_local_idx, predicate_idx, object_local_idx = local_entity_map[s], p, local_entity_map[o]

                    subject_indices.append(subject_local_idx)
                    predicate_indices.append(predicate_idx)
                    object_indices.append(object_local_idx)

                    all_triplet_combinations.append((subject_label, predicate_label, object_label))

        # Skip if no triples
        if subject_indices == [] or predicate_indices == [] or object_indices == []:
            predicted_graph_structures.append([])  # empty graph
            continue

        # Sampling step
        with torch.no_grad():
            sampling_probabilities = torch.tensor([random.random() for _ in range(num_relations * local_num_entities * local_num_entities)])
            sample_mask = sampling_probabilities.bernoulli()

        # Filter the triples, keeping only those that have been sampled as true (indicated by 1 in 'sample_mask')
        sampled_triples = [triple for triple, mask in zip(all_triplet_combinations, sample_mask.tolist()) if mask]
        # 'sampled_triples' contains the subset of triples that have been selected by the model's sampling process, representing the predicted graph structure.

        sampled_graph = list()
        for triples in sampled_triples:
            s, p, o = triples
            sampled_graph.append([s, p, o])
        predicted_graph_structures.append(sampled_graph)

    # Verify the graph predictions by the model
    training_data = post_process_data(training_data, entity_id_to_label=index_to_entity, relation_id_to_label=index_to_relation)
    evaluator = SemanticEvaluator(predicted_graphs=predicted_graph_structures,
                                  ground_truth_graphs=training_data,
                                  rule_checker=rule_check_function,
                                  entity_labels=index_to_entity,
                                  relation_labels=index_to_relation)
    results = evaluator.evaluate_graphs()

    # Report results
    print('Sampling Entities and Structure (SES):')
    print('------------------')
    print('Total graphs: ', len(predicted_graph_structures))
    evaluator.print_results(include_graph_samples=False)
    print('------------------')
    return results


def sample_only_structure(
    test: list = None,
    training_data: list = None,
    index_to_entity: Dict[int, str] = None,
    index_to_relation: Dict[int, str] = None,
    num_relations: Optional[int] = None,
    rule_check_function: Callable = None
) -> Dict:
    """ Sample structure from the model, but use the true entities. """

    assert num_relations is None or isinstance(num_relations, int), "Expected 'num_relations' to be None or an int"
    assert rule_check_function is None or callable(
        rule_check_function), "Expected 'rule_check_function' to be None or a callable"

    predicted_graph_structures = list()

    # Iterate through each graph individually
    for eval_graph in trange(len(test)):
        evaluation_graph = test[eval_graph]

        # Map global indices to local indices for entities
        entities = list(set([triple[0] for triple in evaluation_graph] + [triple[2] for triple in evaluation_graph]))
        local_entity_map = {e: entities.index(e) for e in entities}
        local_num_entities = len(entities)

        subject_indices, predicate_indices, object_indices = [], [], []
        all_triplet_combinations = []  # triples in order with string labels

        # Loop over all possible triples
        for s in entities:
            subject_label = index_to_entity[s]

            for o in entities:
                object_label = index_to_entity[o]

                for p in range(len(index_to_relation)):
                    predicate_label = index_to_relation[p]

                    subject_local_idx, predicate_idx, object_local_idx = local_entity_map[s], p, local_entity_map[o]

                    subject_indices.append(subject_local_idx)
                    predicate_indices.append(predicate_idx)
                    object_indices.append(object_local_idx)

                    all_triplet_combinations.append((subject_label, predicate_label, object_label))

        # Skip if no triples
        if subject_indices == [] or predicate_indices == [] or object_indices == []:
            predicted_graph_structures.append([])  # empty graph
            continue

        # Sampling step
        with torch.no_grad():
            sampling_probabilities = torch.tensor([random.random() for _ in range(num_relations * local_num_entities * local_num_entities)])
            sample_mask = sampling_probabilities.bernoulli()

        # Filter the triples, keeping only those that have been sampled as true (indicated by 1 in 'sample_mask')
        sampled_triples = [triple for triple, mask in zip(all_triplet_combinations, sample_mask.tolist()) if mask]
        # 'sampled_triples' contains the subset of triples that have been selected by the model's sampling process, representing the predicted graph structure.

        sampled_graph = list()
        for triples in sampled_triples:
            s, p, o = triples
            sampled_graph.append([s, p, o])
        predicted_graph_structures.append(sampled_graph)

    # Verify the graph predictions by the model
    training_data = post_process_data(training_data, entity_id_to_label=index_to_entity, relation_id_to_label=index_to_relation)
    evaluator = SemanticEvaluator(predicted_graphs=predicted_graph_structures,
                                  ground_truth_graphs=training_data,
                                  rule_checker=rule_check_function,
                                  entity_labels=index_to_entity,
                                  relation_labels=index_to_relation)
    results = evaluator.evaluate_graphs()

    # Report results
    print('Sampling Structure (SS):')
    print('------------------')
    print('Total graphs: ', len(predicted_graph_structures))
    evaluator.print_results(include_graph_samples=False)
    print('------------------')
    return results


if __name__ == "__main__":
    datasets = ['syn-paths', 'syn-tipr', 'syn-types', 'wd-movies', 'wd-articles']

    for dataset in datasets:
        print(f"Running evaluation on dataset: {dataset}")
        config = {
            "dataset": dataset,  # Name of the dataset
            "final": True,  # Whether to use the final test set
        }
        run_uniform_sampling_evaluation(config)
