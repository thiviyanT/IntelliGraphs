from intelligraphs.baseline_models.knowledge_graph_embedding_model import KGEModel
from intelligraphs.data_loaders import load_data_as_list
from utils import get_device, tic, toc, read_config, save_model
import wandb, math, torch, argparse, random, numpy as np
from tqdm import trange
from intelligraphs.verifier.synthetic import (
    SynPathsVerifier,
    SynTypesVerifier,
    SynTIPRVerifier
)
from intelligraphs.verifier.wikidata import (
    WDArticlesVerifier,
    WDMoviesVerifier
)
from intelligraphs.baseline_models import compute_entity_frequency
from intelligraphs.evaluators import SemanticEvaluator
from intelligraphs.evaluators import post_process_data
from typing import List, Tuple, Dict, Union, Set, Callable, Any


def train(wandb, lmbda=1e-4):
    """ Train baseline models on a dataset """

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device_type)

    config = wandb.config

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

    kge_model = KGEModel(num_entities=num_entities,
                         num_relations=num_relations,
                         embedding_dim=config["emb-size"],
                         biases=config["biases"],
                         edropout=config["edropout"],
                         rdropout=config["rdropout"],
                         decoder=config["decoder"],
                         reciprocal=config["reciprocal"],
                         init_method=config["init_method"])

    # Data parallelism - Use multiple GPUs if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        print(f"Total GPU memory available: "
              f"{round(torch.cuda.get_device_properties(device).total_memory * torch.cuda.device_count() / 1024 ** 3, 1)} GB")
        # print(f"Allocated GPU memory: {round(torch.cuda.memory_allocated(device) / 1024 ** 3, 1)} GB")
        # print(f"Free GPU memory: {round(torch.cuda.memory_reserved(device) / 1024 ** 3, 1)} GB")
        kge_model = torch.nn.DataParallel(kge_model)
    kge_model.to(device)

    opt = torch.optim.Adam(kge_model.parameters(), lr=config["learn-rate"])

    # Weights of the negatives sampled
    if config["loss"] == 'log-loss':
        weight = config["nweight"] if config["nweight"] else None
    else:
        weight = torch.tensor([config["nweight"], 1.0], device=get_device()) if config["nweight"] else None

    # Compute entity frequency from the training data
    entity_frq = compute_entity_frequency(train)

    for epoch in range(config["epochs"]):

        # Evaluate on validation set
        if (epoch == 0) or ((epoch + 1) % config["eval-int"] == 0) or (epoch == config["epochs"] - 1):
            print("Evaluate on validation set")
            evaluate_compression_bits_on_validation_set(
                kge_model=kge_model,
                test=test,
                index_to_relation=index_to_relation,
                entity_frq=entity_frq,
                lmbda=lmbda,
                config=config,
                max_nodes=max_nodes,
                epoch=epoch
            )

        # Training loop
        total_loss = 0.0
        time_forward_pass = time_backward_pass = 0
        time_preparation = time_loss_calculationeparation = 0
        num_batches = 0
        tic()

        kge_model.train(True)

        # Iterate over the training data in batches
        for batch_start_index in trange(0, len(train), config["batch-size"]):
            num_batches += 1
            batch_end_index = min(len(train), batch_start_index + config["batch-size"])

            opt.zero_grad()

            batch = train[batch_start_index:batch_end_index]
            positives = torch.LongTensor(batch).to(get_device())

            # Full-batch negative edges
            assert len(positives.size()) == 3

            # Lists to store positive samples (subject, predicate, object) for the current batch
            subject_positive_list = list()  # Holds subject indices for positive triples
            predicate_positive_list = list()  # Holds predicate indices for positive triples
            object_positive_list = list()  # Holds object indices for positive triples

            # Lists to store negative samples (subject, predicate, object) for the current batch
            subject_negative_list = list()  # Holds subject indices for negative triples
            predicate_negative_list = list()  # Holds predicate indices for negative triples
            object_negative_list = list()  # Holds object indices for negative triples

            for subgraph in positives:
                tic()
                # Map global indices to local indices for entities
                entities = torch.unique(torch.cat([subgraph[:, 0], subgraph[:, 2]])).tolist()
                local_entity_map = {e: entities.index(e) for e in entities}

                subject_positive_indices = torch.tensor([local_entity_map[i] for i in subgraph[:, 0].tolist()])
                predicate_positive_indices = subgraph[:, 1]
                object_positive_indices = torch.tensor([local_entity_map[i] for i in subgraph[:, 2].tolist()])

                num_entities = len(entities)  # Number of unique entities in the subgraph (local)
                num_relations = len(index_to_relation)  # Number of relations in the subgraph (global)

                # construct an adjacency matrix and invert it
                adj = torch.ones(((num_relations, num_entities, num_entities)), dtype=torch.long, device=get_device())
                adj[predicate_positive_indices, subject_positive_indices, object_positive_indices] = 0
                idx = adj.nonzero()

                predicate_negative_indices, subject_negative_indices, object_negative_indices = idx[:, 0], idx[:, 1], idx[:, 2]

                subject_positive_list.append(subject_positive_indices); predicate_positive_list.append(predicate_positive_indices); object_positive_list.append(object_positive_indices)
                subject_negative_list.append(subject_negative_indices); predicate_negative_list.append(predicate_negative_indices); object_negative_list.append(object_negative_indices)

            subject_positive_indices = torch.cat(subject_positive_list, dim=0)
            predicate_positive_indices = torch.cat(predicate_positive_list, dim=0)
            object_positive_indices = torch.cat(object_positive_list, dim=0)
            subject_negative_indices = torch.cat(subject_negative_list, dim=0)
            predicate_negative_indices = torch.cat(predicate_negative_list, dim=0)
            object_negative_indices = torch.cat(object_negative_list, dim=0)

            time_preparation += toc()

            tic()
            pos_scores = kge_model(subject_positive_indices, predicate_positive_indices, object_positive_indices)
            neg_scores = kge_model(subject_negative_indices, predicate_negative_indices, object_negative_indices)
            time_forward_pass += toc()

            tic()

            # Compute bits per graph = -log[ p(S|E) ] - log[ p(E) ]
            # Here we compute the compression bits by taking the negative log-probability of structure and entity

            # Compute -log[ p(S|E) ]
            log_probs_positive = torch.nn.functional.logsigmoid(pos_scores)
            structure_nats_pos = (- log_probs_positive).sum()
            structure_bits_pos = structure_nats_pos / np.log(2)  # nats to bits conversion

            log_probs_negative = torch.nn.functional.logsigmoid(-1.0 * neg_scores)
            structure_nats_neg = (- log_probs_negative).sum()
            structure_bits_neg = structure_nats_neg / np.log(2)  # nats to bits conversion

            # KGE models are training using compression as a learning objective
            # KGE models are required to learn how to reconstruct the structures
            # KGE models are not required to learn how to select entities since they choose entities based on entity frequencies
            loss = structure_bits_pos + (weight * structure_bits_neg)
            loss = loss.mean()

            time_loss_calculationeparation += toc()
            assert not torch.isnan(loss), 'Loss has become NaN'

            total_loss += loss.item()
            wandb.log({"batch_train_compression_loss": loss, "epoch": epoch})

            tic()
            # Initialize the combined loss with the main loss
            loss.backward()
            time_backward_pass += toc()

            # Now we have accumulated the gradients over all subgraphs, so we can step.
            opt.step()

        wandb.log({"epoch_train_compression_loss": total_loss / num_batches,"epoch": epoch})

        if epoch == 0:
            print(f'\n pred: forward {time_forward_pass:.4f}s, backward {time_backward_pass:.4f}s')
            print(f'           prep {time_preparation:.4f}s, loss {time_loss_calculationeparation:.4f}s')
            print(f' total: {toc():.4f}s')

    print('Training finished.')

    # Save parameters of the trained model to file
    save_model(kge_model, f'{config["dataset"]}-{config["decoder"]}', wandb)

    print("Sampling under two settings:")
    print("1) Sampling entities and structure (SES")
    SES_results = sample_entities_structure(
        entity_frq=entity_frq,
        test=test,
        kge_model=kge_model,
        training_data=train,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        rule_check_function=rule_check_function
    )
    print("2) With the true entities and sample only structure (SS)")
    SS_results = sample_only_structure(
        test=test,
        kge_model=kge_model,
        training_data=train,
        index_to_entity=index_to_entity,
        index_to_relation=index_to_relation,
        rule_check_function=rule_check_function
    )

    wandb.log({
        "epoch": epoch,
        "SES_pct_semantics": SES_results['results']['semantics'],
        "SES_pct_novel_semantics": SES_results['results']['novel_semantics'],
        "SES_pct_novel": SES_results['results']['novel'],
        "SES_pct_known": SES_results['results']['known'],
        "SES_pct_empty": SES_results['results']['empty'],
    })
    wandb.log({
        "epoch": epoch,
        "SS_pct_semantics": SS_results['results']['semantics'],
        "SS_pct_novel_semantics": SS_results['results']['novel_semantics'],
        "SS_pct_novel": SS_results['results']['novel'],
        "SS_pct_known": SS_results['results']['known'],
        "SS_pct_empty": SS_results['results']['empty'],
    })


def evaluate_compression_bits_on_validation_set(
    kge_model: torch.nn.Module,
    test: List[List[Tuple[int, int, int]]],
    index_to_relation: Dict[int, str],
    entity_frq: Dict[int, int],
    lmbda: float,
    config: Dict[str, Union[str, int, float, bool, None]],
    max_nodes: int,
    epoch: int
):
    """Computing compression bits for valid/test graphs"""
    kge_model.train(False)

    with torch.no_grad():
        # Compute validation compression bits
        entity_bits = 0
        _pos = list()
        _neg = list()

        # Iterate through each graph individually
        for eval_graph in trange(len(test)):
            evaluation_graph = test[eval_graph]

            # Map global indices to local indices for entities
            entities = list(
                set([triple[0] for triple in evaluation_graph] + [triple[2] for triple in evaluation_graph]))
            local_entity_map = {e: entities.index(e) for e in entities}

            subject_positive_indices = torch.tensor([local_entity_map[triple[0]] for triple in evaluation_graph])
            predicate_positive_indices = torch.tensor([triple[1] for triple in evaluation_graph])
            object_positive_indices = torch.tensor([local_entity_map[triple[2]] for triple in evaluation_graph])

            num_entities = len(entities)  # Number of unique entities in the subgraph (local)
            num_relations = len(index_to_relation)  # Number of relations in the subgraph (global)

            # construct an adjacency matrix and invert it
            adj = torch.ones(((num_relations, num_entities, num_entities)), dtype=torch.long, device=get_device())
            adj[predicate_positive_indices, subject_positive_indices, object_positive_indices] = 0
            idx = adj.nonzero()

            predicate_negative_indices, subject_negative_indices, object_negative_indices = idx[:, 0], idx[:, 1], idx[:, 2]

            pos = kge_model(subject_positive_indices, predicate_positive_indices, object_positive_indices)
            neg = kge_model(subject_negative_indices, predicate_negative_indices, object_negative_indices)

            _pos.append(pos)
            _neg.append(neg)

            sum_e = sum(entity_frq.values())
            for entity in entities:
                # TODO: Ensure we explain why we use relative frequency for prob KGE (i.e. sampling with replacement)
                # TODO: Explain why it would hurt syn-types and syn-tipr. For wikidata, the benefits outweigh the outestimation resulted from sampling with replacement.
                # TODO: Include Laplace smoothing in the kge_model description for the paper
                p_e = (entity_frq[entity] + lmbda) / (sum_e + (lmbda * num_entities))
                entity_bits += - math.log2(p_e)

            # In the sender-receiver framework, one only needs to communicate the number of nodes.
            # We assume that the receiver knows the number of relations.
            if config["dataset"] in ["wd-articles", "wd-movies"]:
                entity_bits += math.log2(max_nodes)

        _pos = torch.cat(_pos, dim=0)
        _neg = torch.cat(_neg, dim=0)

        # Compute bits per graph = -log[ p(S|E) ] - log[ p(E) ]
        # We compute the compression bits by taking the negative log-probability of structure and entity
        # Here we are sampling edges without replacement by nature

        # Compute -log[ p(S|E) ]
        lprobs_pos = torch.nn.functional.logsigmoid(_pos)
        structure_nats_pos = (- lprobs_pos).sum()
        structure_bits_pos = structure_nats_pos / np.log(2)  # nats to bits conversion

        _neg *= -1.0
        lprobs_neg = torch.nn.functional.logsigmoid(_neg)
        structure_nats_neg = (- lprobs_neg).sum()
        structure_bits_neg = structure_nats_neg / np.log(2)  # nats to bits conversion

        structure_bits = structure_bits_pos + structure_bits_neg

        entity_bits = entity_bits / len(test)
        structure_bits = structure_bits / len(test)
        structure_bits = structure_bits.item()  # Convert torch tensor to float

        compression_bits = entity_bits + structure_bits

        wandb.log({
            "valid_compression_bits": compression_bits,
            "valid_structure_bits": structure_bits,
            "valid_entity_bits": entity_bits,
            "epoch": epoch,
        })

        print("Evaluated on after epoch: ", epoch,
              "Entity bits: ", entity_bits,
              "Structure bits: ", structure_bits,
              "Compression bits: ", compression_bits)


def sample_entities_structure(
        entity_frq: Dict[int, int],
        test: List[List[Tuple[int, int, int]]],
        kge_model: torch.nn.Module,
        training_data: List[List[Tuple[int, int, int]]],
        index_to_entity: Dict[int, str],
        index_to_relation: Dict[int, str],
        rule_check_function: Callable[[List[Tuple[str, str, str]]], List[Tuple[str, List[str]]]]
) -> Any:
    """ Sample entities from entity frequency and structure from the kge_model. """
    """ Sample entities from entity frequency and structure from the kge_model. """

    # Compute the probability of each entity in the training set
    p_E = {i: count / sum(entity_frq.values()) for i, count in entity_frq.items()}
    p_E = dict(sorted(p_E.items()))

    predicted_graph_structures = list()

    # Iterate through each graph individually
    for _ in trange(len(test)):

        # Sampling entities based on relative entity frequency
        sampling_probabilities = torch.tensor(list(p_E.values()))
        entity_sampling_mask = sampling_probabilities.bernoulli()
        sampled_entities = [e for e, p in p_E.items() if entity_sampling_mask[e] == 1.0]

        # Map global indices to local indices for entities
        local_entity_map = {e: sampled_entities.index(e) for e in sampled_entities}

        subject_indices, predicate_indices, object_indices = [], [], []
        all_triplet_combinations = []  # triples in order with string labels

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

        with torch.no_grad():
            scores = kge_model(torch.tensor(subject_indices), torch.tensor(predicate_indices), torch.tensor(object_indices))
            sampling_probabilities = torch.sigmoid(scores)
            sample_mask = sampling_probabilities.bernoulli()

        # Filter the triples, keeping only those that have been sampled as true (indicated by 1 in 'sample_mask')
        sampled_triples = [t for t, m in zip(all_triplet_combinations, sample_mask.tolist()) if m]
        # 'sampled_triples' contains the subset of triples that have been selected by the model's sampling process, representing the predicted graph structure.

        sampled_graph = list()
        for all_triplet_combinations in sampled_triples:
            s, p, o = all_triplet_combinations
            sampled_graph.append([s, p, o])
        predicted_graph_structures.append(sampled_graph)

    # Verify the graph predictions by the kge_model
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
    evaluator.print_results()
    print('------------------')
    return results


def sample_only_structure(
    test: List[List[Tuple[int, int, int]]],
    kge_model: torch.nn.Module,
    training_data: List[List[Tuple[int, int, int]]],
    index_to_entity: Dict[int, str],
    index_to_relation: Dict[int, str],
    rule_check_function: Callable[[List[Tuple[str, str, str]]], List[Tuple[str, List[str]]]]
) -> Any:
    """ Sample structure from the kge_model, but use the true entities. """

    predicted_graph_structures = list()

    # Iterate through each graph individually
    for eval_graph in trange(len(test)):
        evaluation_graph = test[eval_graph]

        # Map global indices to local indices for entities
        entities = list(set([triple[0] for triple in evaluation_graph] + [triple[2] for triple in evaluation_graph]))
        local_entity_map = {e: entities.index(e) for e in entities}

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

        with torch.no_grad():
            scores = kge_model(torch.tensor(subject_indices), torch.tensor(predicate_indices), torch.tensor(object_indices))
            sampling_probabilities = torch.sigmoid(scores)
            sample_mask = sampling_probabilities.bernoulli()

        # Filter the triples, keeping only those that have been sampled as true (indicated by 1 in 'sample_mask')
        sampled_triples = [t for t, m in zip(all_triplet_combinations, sample_mask.tolist()) if m]
        # 'sampled_triples' contains the subset of triples that have been selected by the model's sampling process, representing the predicted graph structure.

        sampled_graph = list()
        for all_triplet_combinations in sampled_triples:
            s, p, o = all_triplet_combinations
            sampled_graph.append([s, p, o])
        predicted_graph_structures.append(sampled_graph)

    # Verify the graph predictions by the kge_model
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
    evaluator.print_results()
    print('------------------')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="filepath to configurations")
    args = parser.parse_args()

    # Default configuration
    hyperparameter_defaults = {
        "epochs": 150,  # Number of epochs to train
        "dataset": 'syn-paths',  # Name of the dataset
        "decoder": 'transe',  # Decoder to use
        "eval-size": None,  # Number of graphs to evaluate on
        "eval-int": 20,  # Evaluate every eval-int epochs
        "batch-size": 32,  # Batch size
        "emb-size": 128,  # Embedding size
        "learn-rate": 0.0001,  # Learning rate
        "final": True,  # Whether to use final regularization
        "loss": 'log-loss',  # Loss function to use
        "nweight": 1.0,  # Weight for negative samples
        "biases": True,  # Whether to use biases
        "edropout": None,  # Entity dropout
        "rdropout": None,  # Relation dropout
        "reciprocal": False,  # Whether to use reciprocal relations
        "init_method": 'uniform',  # Initialization method
    }

    # Initialize wandb
    wandb.init()

    # Set default hyperparameters in wandb.config
    wandb.config.update(hyperparameter_defaults, allow_val_change=True)

    # Override with values from the config file if provided
    if args.config is not None:
        my_yaml_file = read_config(args.config)
        wandb.config.update(my_yaml_file, allow_val_change=True)

    print('Hyper-parameters: ', wandb.config)
    train(wandb)
