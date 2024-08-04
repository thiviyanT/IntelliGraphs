from _context import baselines
from util.semantic_checker import check_semantics, check_pathgraph, check_academic_graph, check_location_graph, check_movie_graph, check_article_graph
from util import d, tic, toc, get_slug, compute_entity_frequency, read_config
from tqdm import trange
import multiprocessing as mp
import wandb
import torch
import random


def train(wandb):
    """ Train baseline models on a dataset """

    config = wandb.config

    train, val, test, (n2i, i2n), (r2i, i2r) = \
        baselines.load(config["dataset"], padding=True)

    if config["final"]:
        train, test = torch.cat([train, val], dim=0), test
    else:
        train, test = train, val

    print(len(i2n), 'nodes')
    print(len(i2r), 'relations')
    print(train.size(0), 'training triples')
    print(test.size(0), 'test triples')
    print(train.size(0) + test.size(0), 'total triples')

    num_entities = len(i2n)
    num_relations = len(i2r)

    # Sampling under two settings:

    # 1) With the sampled entities
    results = sample_entities_structure(test, train, i2n, i2r, config, num_entities, num_relations)
    print({
        "SEE_pct_semantics": results["pct_semantics"],
        "SEE_pct_valid_but_wrong_length": results["pct_valid_but_wrong_length"],
        "SEE_pct_novel_semantics": results["pct_novel_semantics"],
        "SEE_pct_novel": results["pct_novel"],
        "SEE_pct_original": results["pct_original"],
        "SEE_pct_empty": results["pct_empty"],
    })

    # 2) Sample structure only
    results = sample_structure(test, train, i2n, i2r, config, num_entities, num_relations)
    print({
        "SS_pct_semantics": results["pct_semantics"],
        "SS_pct_valid_but_wrong_length": results["pct_valid_but_wrong_length"],
        "SS_pct_novel_semantics": results["pct_novel_semantics"],
        "SS_pct_novel": results["pct_novel"],
        "SS_pct_original": results["pct_original"],
        "SS_pct_empty": results["pct_empty"],
    })


def sample_entities_structure(test, training_data, i2n, i2r, config, num_entities, num_relations):
    """ Sample entities from entity frequency and structure from the model. """

    if config["dataset"] == "syn-paths":
        semantic_func = check_pathgraph
        expected_graph_length = 3
    elif config["dataset"] == "syn-tipr":
        semantic_func = check_academic_graph
        expected_graph_length = 5
    elif config["dataset"] == "syn-types":
        semantic_func = check_location_graph
        expected_graph_length = 3
    elif config["dataset"] == "wd-movies":
        semantic_func = check_movie_graph
        expected_graph_length = None  # No size constraint
    elif config["dataset"] == "wd-articles":
        semantic_func = check_article_graph
        expected_graph_length = None  # No size constraint
    else:
        raise NotImplementedError('No check implemented for this dataset')

    model_output = list()

    i2n = {i: n for i, n in enumerate(i2n)}

    # -- Use all graphs in the validation set
    testsub = test

    # Batched evaluation
    for fr in trange(0, testsub.size(0), config["batch-size"]):

        to = min(testsub.size(0), fr + config["batch-size"])
        eval_graphs = testsub[fr:to].to(d())

        for b, subgraph in enumerate(eval_graphs):

            if wandb.config["padding"]:
                subgraph = subgraph[subgraph[:, 1] != -1]  # Remove padding

            # Map global indices to local indices for entities
            probs = torch.tensor([random.random() for _ in range(num_entities)])
            entities_sample = probs.bernoulli()
            sampled_entities = [e for e, p in i2n.items() if entities_sample[e] == 1.0]
            entity_map_ = {e: sampled_entities.index(e) for e in sampled_entities}
            number_nodes = len(sampled_entities)

            ss, ps, os = [], [], []
            triples = []  # triples in order with string labels

            # Loop over all possible triples
            for s in sampled_entities:
                sstr = i2n[s]

                for o in sampled_entities:
                    ostr = i2n[o]

                    for p in range(len(i2r)):
                        pstr = i2r[p]

                        skge, pkge, okge = entity_map_[s], p, entity_map_[o]

                        ss.append(skge)
                        ps.append(pkge)
                        os.append(okge)

                        triples.append((sstr, pstr, ostr))

            # Skip if no triples
            if ss == [] or ps == [] or os == []:
                model_output.append([])  # empty graph
                continue

            with torch.no_grad():
                probs = torch.tensor([random.random() for _ in range(num_relations * number_nodes * number_nodes)])
                sample = probs.bernoulli()

            true_triples = [t for t, m in zip(triples, sample.tolist()) if m]

            y = list()
            for triples in true_triples:
                s, p, o = triples
                y.append([s, p, o])
            model_output.append(y)

    # Run verification
    results, graphs = check_semantics(model_output,
                                      training_data.tolist(),
                                      semantic_func,
                                      entity_labels=i2n,
                                      relation_labels=i2r,
                                      graph_size=expected_graph_length)

    # Report results
    # print('Sampling Structure (SE):')
    # print('------------------')
    # print('Total graphs: ', len(model_output))
    # print(results)
    # print('Valid graphs: ', graphs['valid_graphs'][:50])
    # print('')
    # print('Valid Novel graphs', graphs['valid_novel_graphs'][:50])
    # print('')
    # print('Invalid graphs: ', graphs['invalid_graphs'][:50])
    # print('------------------')
    return results


def sample_structure(test, training_data, i2n, i2r, config, num_entities, num_relations):
    """ Sample structure from the model, but use the true entities. """

    if config["dataset"] == "syn-paths":
        semantic_func = check_pathgraph
        expected_graph_length = 3
    elif config["dataset"] == "syn-tipr":
        semantic_func = check_academic_graph
        expected_graph_length = 5
    elif config["dataset"] == "syn-types":
        semantic_func = check_location_graph
        expected_graph_length = 3
    elif config["dataset"] == "wd-movies":
        semantic_func = check_movie_graph
        expected_graph_length = None  # No size constraint
    elif config["dataset"] == "wd-articles":
        semantic_func = check_article_graph
        expected_graph_length = None  # No size constraint
    else:
        raise NotImplementedError('No check implemented for this dataset')

    model_output = list()

    # -- Use all graphs in the validation set
    testsub = test

    # Batched evaluation
    for fr in trange(0, testsub.size(0), config["batch-size"]):

        to = min(testsub.size(0), fr + config["batch-size"])
        eval_graphs = testsub[fr:to].to(d())

        for b, subgraph in enumerate(eval_graphs):

            if wandb.config["padding"]:
                subgraph = subgraph[subgraph[:, 1] != -1]  # Remove padding

            # Map global indices to local indices for entities
            entities = torch.unique(torch.cat([subgraph[:, 0], subgraph[:, 2]])).tolist()
            entity_map_ = {e: entities.index(e) for e in entities}
            number_nodes = len(entities)

            ss, ps, os = [], [], []
            triples = []  # triples in order with string labels

            # Loop over all possible triples
            for s in entities:
                sstr = i2n[s]

                for o in entities:
                    ostr = i2n[o]

                    for p in range(len(i2r)):
                        pstr = i2r[p]

                        skge, pkge, okge = entity_map_[s], p, entity_map_[o]

                        ss.append(skge)
                        ps.append(pkge)
                        os.append(okge)

                        triples.append((sstr, pstr, ostr))

            # Skip if no triples
            if ss == [] or ps == [] or os == []:
                model_output.append([])  # empty graph
                continue

            with torch.no_grad():
                probs = torch.tensor([random.random() for _ in range(num_relations * number_nodes * number_nodes)])
                sample = probs.bernoulli()

            true_triples = [t for t, m in zip(triples, sample.tolist()) if m]
            # print(true_triples)
            # print(triples)

            y = list()
            for triples in true_triples:
                s, p, o = triples
                y.append([s, p, o])
            model_output.append(y)

    # Run verification
    results, graphs = check_semantics(model_output,
                                      training_data.tolist(),
                                      semantic_func,
                                      entity_labels=i2n,
                                      relation_labels=i2r,
                                      graph_size=expected_graph_length)

    # Report results
    # print('Sampling Structure (SE):')
    # print('------------------')
    # print('Total graphs: ', len(model_output))
    # print(results)
    # print('Valid graphs: ', graphs['valid_graphs'][:50])
    # print('')
    # print('Valid Novel graphs', graphs['valid_novel_graphs'][:50])
    # print('')
    # print('Invalid graphs: ', graphs['invalid_graphs'][:50])
    # print('------------------')
    return results


if __name__ == "__main__":

    mp.set_start_method('spawn')

    # Default configuration
    hyperparameter_defaults = {
        "dataset": 'syn-paths',  # Name of the dataset
        "final": False,  # Whether to use the final test set
        "batch-size": 1024,  # Batch size
        "padding": True,  # Whether to use padding
    }

    wandb.init(
        project="kgi",
        entity="nesy-gems",
        name=f"Sampling Test",
        notes="",
        tags=['sampling'],
        config=hyperparameter_defaults,
    )

    print('Hyper-parameters: ', wandb.config)
    train(wandb)
