import math
from _context import models
from util import d, tic, toc, get_slug, compute_entity_frequency, read_config
from util.semantic_checker import check_semantics, check_pathgraph, check_academic_graph, check_location_graph, check_movie_graph, check_article_graph, save_model
import torch.nn.functional as F
from tqdm import trange
import random
import multiprocessing as mp
import wandb
import numpy as np
import torch
import argparse


def train(wandb, lmbda=1e-4):
    """ Train baseline models on a dataset """

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', dev)

    config = wandb.config

    train, val, test, (n2i, i2n), (r2i, i2r), (min_edges, max_edges), (min_nodes, max_nodes) = \
        models.load_data(config["dataset"], padding=config["padding"])

    if config["final"]:
        train, test = torch.cat([train, val], dim=0), test
    else:
        train, test = train, val

    print(len(i2n), 'nodes')
    print(len(i2r), 'relations')
    print(train.size(0), 'training triples')
    print(test.size(0), 'test triples')
    print(train.size(0) + test.size(0), 'total triples')

    model = models.KGEModel(
        n=len(i2n), r=len(i2r), embedding=config["emb-size"], biases=config["biases"],
        edropout=config["edropout"], rdropout=config["rdropout"], decoder=config["decoder"],
        reciprocal=config["reciprocal"], init_method=config["init_method"])

    # Data parallelism - Use multiple GPUs if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        print(f"Total GPU memory available: "
              f"{round(torch.cuda.get_device_properties(device).total_memory * torch.cuda.device_count() / 1024 ** 3, 1)} GB")
        # print(f"Allocated GPU memory: {round(torch.cuda.memory_allocated(device) / 1024 ** 3, 1)} GB")
        # print(f"Free GPU memory: {round(torch.cuda.memory_reserved(device) / 1024 ** 3, 1)} GB")
        model = torch.nn.DataParallel(model)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config["learn-rate"])

    # nr of negatives sampled
    if config["loss"] == 'log-loss':
        weight = config["nweight"] if config["nweight"] else None
    else:
        weight = torch.tensor([config["nweight"], 1.0], device=d()) if config["nweight"] else None

    # Compute entity frequency from the training data
    frq = compute_entity_frequency(train)

    for e in range(config["epochs"]):

        # Evaluate on validation set
        if ((e + 1) % config["eval-int"] == 0) or e == config["epochs"] - 1:
            with torch.no_grad():

                model.train(False)

                #######################################################################################################
                # Computing compression bits for valid/test graphs
                #######################################################################################################

                if config["eval-size"] is None:
                    # -- Use all graphs in the validation set
                    testsub = test
                else:
                    # -- Use a random subset of graphs in the validation set
                    testsub = test[random.sample(range(test.size(0)), k=config["eval-size"])]

                # Compute validation compression bits
                entity_bits = 0
                valid_compression_bits = 0
                _pos = list()
                _neg = list()

                # Batched evaluation
                for fr in trange(0, testsub.size(0), config["batch-size"]):

                    to = min(testsub.size(0), fr + config["batch-size"])
                    eval_graphs = testsub[fr:to].to(d())

                    for b, padded_subgraph in enumerate(eval_graphs):

                        # Filter out rows that are not equal to the padding triple
                        padding_triple = torch.tensor([-1, -1, -1])
                        subgraph = padded_subgraph[~torch.all(padded_subgraph == padding_triple, dim=1)]

                        # Map global indices to local indices for entities
                        entities = torch.unique(torch.cat([subgraph[:, 0], subgraph[:, 2]])).tolist()
                        entity_map_ = {e: entities.index(e) for e in entities}

                        s_pos = torch.tensor([entity_map_[i] for i in subgraph[:, 0].tolist()])
                        p_pos = subgraph[:, 1]
                        o_pos = torch.tensor([entity_map_[i] for i in subgraph[:, 2].tolist()])

                        num_entities = len(entities)  # Number of unique entities in the subgraph (local)
                        num_relations = len(i2r)  # Number of relations in the subgraph (global)
                        num_edges = subgraph.size(0)

                        # construct an adjacency matrix and invert it
                        adj = torch.ones(((num_relations, num_entities, num_entities)), dtype=torch.long, device=d())
                        adj[p_pos, s_pos, o_pos] = 0
                        idx = adj.nonzero()

                        p_neg, s_neg, o_neg = idx[:, 0], idx[:, 1], idx[:, 2]

                        pos = model(s_pos, p_pos, o_pos)
                        neg = model(s_neg, p_neg, o_neg)

                        _pos.append(pos)
                        _neg.append(neg)

                        sum_e = sum(frq.values())
                        for entity in entities:
                            # TODO: Ensure we explain why we use relative frequency for prob KGE (i.e. sampling with replacement)
                            # TODO: Explain why it would hurt syn-types and syn-tipr. For wikidata, the benefits outweigh the outestimation resulted from sampling with replacement.
                            # TODO: Include Laplace smoothing in the model description for the paper
                            p_e = (frq[entity] + lmbda) / (sum_e + (lmbda * num_entities))
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
                lprobs_pos = F.logsigmoid(_pos)
                structure_nats_pos = (- lprobs_pos).sum()
                structure_bits_pos = structure_nats_pos / np.log(2)  # nats to bits conversion

                _neg *= -1.0
                lprobs_neg = F.logsigmoid(_neg)
                structure_nats_neg = (- lprobs_neg).sum()
                structure_bits_neg = structure_nats_neg / np.log(2)  # nats to bits conversion

                structure_bits = structure_bits_pos + structure_bits_neg

                entity_bits = entity_bits / testsub.size(0)
                structure_bits = structure_bits / testsub.size(0)
                structure_bits = structure_bits.item()  # Convert torch tensor to float

                compression_bits = entity_bits + structure_bits

                wandb.log({
                    "valid_compression_bits": compression_bits,
                    "valid_structure_bits": structure_bits,
                    "valid_entity_bits": entity_bits,
                    "epoch": e,
                })

                print("Evaluated on after epoch: ", e,
                      "Entity bits: ", entity_bits,
                      "Structure bits: ", structure_bits,
                      "Compression bits: ", compression_bits)

        # Training loop
        seeni, sumloss = 0, 0.0
        tforward = tbackward = 0
        rforward = rbackward = 0
        tprep = tloss = 0
        num_batches = 0
        tic()

        for fr in trange(0, train.size(0), config["batch-size"]):
            num_batches += 1
            to = min(train.size(0), fr + config["batch-size"])

            model.train(True)

            opt.zero_grad()
            positives = train[fr:to].to(d())

            # Full-batch negative edges
            assert len(positives.size()) == 3
            bs, _, _ = positives.size()

            _s_pos = list(); _p_pos = list(); _o_pos = list()
            _s_neg = list(); _p_neg = list(); _o_neg = list()

            for b, padded_subgraph in enumerate(positives):

                # Filter out rows that are not equal to the padding triple
                padding_triple = torch.tensor([-1, -1, -1])
                subgraph = padded_subgraph[~torch.all(padded_subgraph == padding_triple, dim=1)]

                tic()

                # Map global indices to local indices for entities
                entities = torch.unique(torch.cat([subgraph[:, 0], subgraph[:, 2]])).tolist()
                entity_map_ = {e: entities.index(e) for e in entities}

                s_pos = torch.tensor([entity_map_[i] for i in subgraph[:, 0].tolist()])
                p_pos = subgraph[:, 1]
                o_pos = torch.tensor([entity_map_[i] for i in subgraph[:, 2].tolist()])

                num_entities = len(entities)  # Number of unique entities in the subgraph (local)
                num_relations = len(i2r)  # Number of relations in the subgraph (global)

                # construct an adjacency matrix and invert it
                adj = torch.ones(((num_relations, num_entities, num_entities)), dtype=torch.long, device=d())
                adj[p_pos, s_pos, o_pos] = 0
                idx = adj.nonzero()

                p_neg, s_neg, o_neg = idx[:, 0], idx[:, 1], idx[:, 2]

                _s_pos.append(s_pos); _p_pos.append(p_pos); _o_pos.append(o_pos)
                _s_neg.append(s_neg); _p_neg.append(p_neg); _o_neg.append(o_neg)

            s_pos = torch.cat(_s_pos, dim=0)
            p_pos = torch.cat(_p_pos, dim=0)
            o_pos = torch.cat(_o_pos, dim=0)
            s_neg = torch.cat(_s_neg, dim=0)
            p_neg = torch.cat(_p_neg, dim=0)
            o_neg = torch.cat(_o_neg, dim=0)

            tprep += toc()

            tic()
            pos_scores = model(s_pos, p_pos, o_pos)
            neg_scores = model(s_neg, p_neg, o_neg)
            tforward += toc()

            tic()

            # Compute bits per graph = -log[ p(S|E) ] - log[ p(E) ]
            # Here we compute the compression bits by taking the negative log-probability of structure and entity

            # Compute -log[ p(S|E) ]
            lprobs_pos = F.logsigmoid(pos_scores)
            structure_nats_pos = (- lprobs_pos).sum()
            structure_bits_pos = structure_nats_pos / np.log(2)  # nats to bits conversion

            lprobs_neg = F.logsigmoid(-1.0 * neg_scores)
            structure_nats_neg = (- lprobs_neg).sum()
            structure_bits_neg = structure_nats_neg / np.log(2)  # nats to bits conversion

            # KGE models are training using compression as a learning objective
            # KGE models are required to learn how to reconstruct the structures
            # KGE models are not required to learn how to select entities since they choose entities based on entity frequencies
            loss = structure_bits_pos + (weight * structure_bits_neg)
            loss = loss.mean()

            tloss += toc()
            assert not torch.isnan(loss), 'Loss has become NaN'

            wandb.log(
                {
                    "batch_train_compression_loss": loss,
                    "epoch": e,
                }
            )

            tic()
            # Initialize the combined loss with the main loss
            combined_loss = loss

            if config["reg-eweight"] is not None:
                e_regloss = model.penalty(which='entities', p=config["reg-exp"], rweight=config["reg-eweight"])
                combined_loss += e_regloss  # Add entity regularization loss to the combined loss

            if config["reg-rweight"] is not None:
                r_regloss = model.penalty(which='relations', p=config["reg-exp"], rweight=config["reg-rweight"])
                combined_loss += r_regloss  # Add relation regularization loss to the combined loss
            rforward += toc()

            # Backward pass for the combined loss
            tic()
            combined_loss.backward()  # Compute gradients for the combined loss
            rbackward += toc()

            # Now we have accumulated the gradients over all subgraphs, so we can step.
            opt.step()

        wandb.log(
            {
                "epoch_train_compression_loss": sumloss / num_batches,
                "epoch": e,
             }
        )

        if e == 0:
            print(f'\n pred: forward {tforward:.4f}s, backward {tbackward:.4f}s')
            print(f'   reg: forward {rforward:.4f}s, backward {rbackward:.4f}s')
            print(f'           prep {tprep:.4f}s, loss {tloss:.4f}s')
            print(f' total: {toc():.4f}')
            # -- NB: these numbers will not be accurate for GPU runs unless CUDA_LAUNCH_BLOCKING is set to 1

    print('Training finished.')

    save_model(model, f'{config["dataset"]}-{config["decoder"]}', wandb)

    # 1) With the sampled entities
    print("Sampling with entities and structure")
    results = sample_entities_structure(frq, test, model, train, i2n, i2r, config)

    wandb.log({
        "epoch": e,
        "SEE_pct_semantics": results["pct_semantics"],
        "SEE_pct_valid_but_wrong_length": results["pct_valid_but_wrong_length"],
        "SEE_pct_novel_semantics": results["pct_novel_semantics"],
        "SEE_pct_novel": results["pct_novel"],
        "SEE_pct_original": results["pct_original"],
        "SEE_pct_empty": results["pct_empty"],
    })

    # 2) With the true entities
    print("Sampling with structure only")
    results = sample_structure(frq, test, model, train, i2n, i2r, config)
    wandb.log({
        "epoch": e,
        "SS_pct_semantics": results["pct_semantics"],
        "SS_pct_valid_but_wrong_length": results["pct_valid_but_wrong_length"],
        "SS_pct_novel_semantics": results["pct_novel_semantics"],
        "SS_pct_novel": results["pct_novel"],
        "SS_pct_original": results["pct_original"],
        "SS_pct_empty": results["pct_empty"],
    })


def sample_entities_structure(frq, test, model, training_data, i2n, i2r, config):
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

    # compute the probability of each entity in the training set
    p_E = {i: count / sum(frq.values()) for i, count in frq.items()}
    p_E = dict(sorted(p_E.items()))
    # print("Sampling entities from the distribution: ", p_E)

    if config["eval-size"] is None:
        # -- Use all graphs in the validation set
        testsub = test
    else:
        # -- Use a random subset of graphs in the validation set
        testsub = test[random.sample(range(test.size(0)), k=config["eval-size"])]

    # Batched evaluation
    for fr in trange(0, testsub.size(0), config["batch-size"]):

        to = min(testsub.size(0), fr + config["batch-size"])
        eval_graphs = testsub[fr:to].to(d())

        for b, _ in enumerate(eval_graphs):

            # Map global indices to local indices for entities
            probs = torch.tensor(list(p_E.values()))
            entities_sample = probs.bernoulli()
            sampled_entities = [e for e, p in p_E.items() if entities_sample[e] == 1.0]
            entity_map_ = {e: sampled_entities.index(e) for e in sampled_entities}

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
                scores = model(torch.tensor(ss), torch.tensor(ps), torch.tensor(os))
                probs = torch.sigmoid(scores)

                structure_sample = probs.bernoulli()

            true_triples = [t for t, m in zip(triples, structure_sample.tolist()) if m]

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
    print('Sampling Entities and Structure (SEE):')
    print('------------------')
    print('Total graphs: ', len(model_output))
    print(results)
    print('Valid graphs: ', graphs['valid_graphs'][:50])
    print('')
    print('Valid Novel graphs', graphs['valid_novel_graphs'][:50])
    print('')
    print('Invalid graphs: ', graphs['invalid_graphs'][:50])
    print('------------------')
    return results


def sample_structure(frq, test, model, training_data, i2n, i2r, config):
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

    if config["eval-size"] is None:
        # -- Use all graphs in the validation set
        testsub = test
    else:
        # -- Use a random subset of graphs in the validation set
        testsub = test[random.sample(range(test.size(0)), k=config["eval-size"])]

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
                scores = model(torch.tensor(ss), torch.tensor(ps), torch.tensor(os))
                probs = torch.sigmoid(scores)
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
    print('Sampling Structure (SE):')
    print('------------------')
    print('Total graphs: ', len(model_output))
    print(results)
    print('Valid graphs: ', graphs['valid_graphs'][:50])
    print('')
    print('Valid Novel graphs', graphs['valid_novel_graphs'][:50])
    print('')
    print('Invalid graphs: ', graphs['invalid_graphs'][:50])
    print('------------------')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="filepath to configurations")
    args = parser.parse_args()

    mp.set_start_method('spawn')

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
        "reg-exp": 2,  # Regularization exponent
        "reg-eweight": None,  # Regularization weight for entity embeddings
        "reg-rweight": 0.1,  # Regularization weight for relation embeddings
        "final": True,  # Whether to use final regularization
        "loss": 'log-loss',  # Loss function to use
        "negative-sampling-strategy": 'matrix inversion',  # Negative sampling strategy
        "nweight": 1.0,  # Weight for negative samples
        "biases": True,  # Whether to use biases
        "edropout": None,  # Entity dropout
        "rdropout": None,  # Relation dropout
        "reciprocal": False,  # Whether to use reciprocal relations
        "init_method": 'uniform',  # Initialization method
        "padding": True,  # Whether to use padding
    }

    # Initialize wandb
    wandb.init(mode="disabled")

    # Set default hyperparameters in wandb.config
    wandb.config.update(hyperparameter_defaults, allow_val_change=True)

    # Override with values from the config file if provided
    if args.config is not None:
        my_yaml_file = read_config(args.config)
        wandb.config.update(my_yaml_file, allow_val_change=True)

    print('Hyper-parameters: ', wandb.config)
    train(wandb)
