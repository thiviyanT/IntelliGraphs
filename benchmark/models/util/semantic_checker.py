from graphviz import Digraph
import torch
import os
import random
from tqdm import trange
from pprint import pprint


def get_sample_sizes(sample_count, batch_size):
    i = []
    remainder = sample_count % batch_size
    i.append(remainder)
    _sample_count = sample_count - remainder
    x = int(_sample_count / batch_size)
    i += [batch_size] * x
    assert sum(i) == sample_count
    i.sort(reverse=True)
    return i


def list_to_dictionary(obj):
    """ Some trivial preprocessing steps """
    if type(obj) is list:
        return {i: v for i, v in enumerate(obj)}
    elif type(obj) is dict:
        return {int(k): v for k, v in obj.items()}
    else:
        raise TypeError


def print_results(config, recon_rslt, smpl_rslt, final=False):
    """ Print reconstruction and sampling results """
    if final:
        print('\n\n')
        print('=' * 100)
        print(f'Experiment: {config["wandb"]["name"]}')
        print(f'Notes: {config["wandb"]["notes"]}')
    else:
        print('=-' * 50)
    print(f"Reconstruction Results:")
    pprint(recon_rslt)
    print(f"Sampling Results:")
    pprint(smpl_rslt)


def label_triples(graphs, i2e, i2r, null_entity):
    """
    Add label to triples

    Assumption: the last entity is null-entity
    """
    labelled_triples = list()
    for triples in graphs:
        y = list()
        for triple in triples:
            s, p, o = triple
            if s == null_entity or o == null_entity:
                continue
            y.append((i2e[s], i2r[p], i2e[o]))
        labelled_triples.append(frozenset(y))
    return labelled_triples


def print_graphs(graphs, limit=1):
    print('='*100)
    for graph in graphs[:limit]:
        print('digraph G {')
        for triple in graph:
            s, p, o = triple
            print(f' "{s}" -> "{o}" [ label="{p}" ];')
        print('}')
        print('\n')
    print('=' * 100)


def check_semantics(model_output, data, func, entity_labels=None, relation_labels=None, graph_size=None):
    """ Checks the semantics of decoded latent structures """
    valid = 0  # Number of valid graphs
    valid_graphs = []  # List of valid graphs
    valid_novel = 0  # Number of valid graphs that are not present in the training data
    valid_novel_graphs = []  # List of valid graphs that are not present in the training data
    valid_but_wrong_length = 0  # Number of graphs that are valid but have a different length than the target graph
    novel_structure = 0  # Number of graphs that are not present in the training data
    original_structure = 0  # Number of graphs that are present in the training data
    empty_graph = 0  # Number of empty graphs
    invalid_graphs = []  # List of invalid graphs

    graph_size = graph_size

    # Iterate through the generated samples
    for i in trange(len(model_output)):

        # Check if the graph is empty. If so, skip the other tests
        if len(model_output[i]) == 0:
            empty_graph += 1
            continue

        # Check if data present in the training data
        if model_output[i] in data:
            original_structure += 1

            # Check the semantic validity and the graph size
            if func(model_output[i], entity_labels, relation_labels, length=graph_size):
                valid += 1
                valid_graphs.append(model_output[i])
            else:
                # Just check the semantic validity
                if func(model_output[i], entity_labels, relation_labels, length=None):
                    valid_but_wrong_length += 1
                    valid_graphs.append(model_output[i])
                else:
                    invalid_graphs.append(model_output[i])
        else:
            novel_structure += 1

            # Check the semantic validity and the graph size
            if func(model_output[i], entity_labels, relation_labels, length=graph_size):
                valid_novel += 1
                valid += 1
                valid_novel_graphs.append(model_output[i])
            else:
                # Just check the semantic validity
                if func(model_output[i], entity_labels, relation_labels, length=None):
                    valid_but_wrong_length += 1
                    valid_novel_graphs.append(model_output[i])
                else:
                    invalid_graphs.append(model_output[i])

    pct_semantics = round((valid / len(model_output)) * 100, 2)
    pct_novel_semantics = round((valid_novel / len(model_output)) * 100, 2)
    pct_novel = round((novel_structure / len(model_output)) * 100, 2)
    pct_original = round((original_structure / len(model_output)) * 100, 2)
    pct_empty = round((empty_graph / len(model_output)) * 100, 2)
    pct_valid_but_wrong_length = round((valid_but_wrong_length / len(model_output)) * 100, 2)
    return ({
        'pct_semantics': pct_semantics,
        'pct_valid_but_wrong_length': pct_valid_but_wrong_length,
        'pct_novel_semantics': pct_novel_semantics,
        'pct_novel': pct_novel,
        'pct_original': pct_original,
        'pct_empty': pct_empty
    }, {
        'valid_graphs': valid_graphs,
        'valid_novel_graphs': valid_novel_graphs,
        'invalid_graphs': invalid_graphs,
    })


def print_graph(graph, i2n, i2r, view=False, name='input_graph'):
    """ Prints graphs using graphviz """
    dot = Digraph(format='png')
    for triple in graph:
        s, p, o = triple

        dot.node(i2n[s])
        dot.node(i2n[o])
        dot.edge(i2n[s], i2n[o], label=i2r[p])

    dot.render(f'graph-sample/{name}.dot', view=view)


def triples2adj(triples, num_r, num_n, value=1, filter_padding=False, device='cpu'):
    """ Converts triples into an adjacency matrices using torch operations """

    """
    Note: It is quicker to filter out padding edges after adjacency matrix has been created
    """
    if filter_padding:
        num_r_plus = num_r + 1

    # Check if dataset is batched
    if len(triples.shape) == 3:
        batch_size = triples.shape[0]
        subjects = triples[:, :, 0]
        predicates = triples[:, :, 1]
        objects = triples[:, :, 2]

        num_triples = triples.shape[1]
        i = torch.arange(batch_size)[None, :].t().expand(-1, num_triples)

        if filter_padding:
            x = torch.zeros(batch_size, num_r_plus, num_n, num_n)
        else:
            x = torch.zeros(batch_size, num_r, num_n, num_n)

        x[i, predicates, subjects, objects] = value

        # Filter out triples used for padding
        if filter_padding:
            x = x[:, 0:num_r, :, :]
    else:
        subjects = triples[:, 0]
        predicates = triples[:, 1]
        objects = triples[:, 2]

        if filter_padding:
            x = torch.zeros(num_r_plus, num_n, num_n)
        else:
            x = torch.zeros(num_r, num_n, num_n)

        x[predicates, subjects, objects] = value

        # Filter out triples used for padding
        if filter_padding:
            x = x[0:num_r, :, :]
    return x.to(device)


def save_model(model, name, wandb):
    """ Save model to Weights&Biases """
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{name}.pt'))
    print('Model saved to Weights&Biases')
    print(os.path.join(wandb.run.dir, f'{name}.pt'))


def load_model(name):
    """ Load saved model """
    return torch.load(f'{name}', map_location=torch.device('cpu'))


class Graph:
    """
    Check graph connectivity.

    Algorithm:
    Starting from a random node, perform a depth first-search and mark visited nodes.
    If the number of visited nodes is the same as the number of nodes in the graph, then it's a connected graph.
    """
    def __init__(self, triples):
        super(Graph, self).__init__()
        self.graph = dict()
        self.nodes = dict()  # Stores booleans for visited nodes
        for triple in triples:
            s, p, o = triple
            if s not in self.nodes:
                self.nodes[s] = False
            if o not in self.nodes:
                self.nodes[o] = False
            self.add_edge(s, o)

    def add_edge(self, source, target):
        """ Add edges to graph """

        # Add forward link
        if source not in self.graph:
            self.graph[source] = []
        self.graph[source].append(target)

        # Add backward link for undirected graph
        if target not in self.graph:
            self.graph[target] = []
        self.graph[target].append(source)

    def dfs(self, x):
        """ Depth-First Search and check visited vertices """
        self.nodes[x] = True
        if x not in self.graph:
            self.graph[x] = {}

        for i in self.graph[x]:
            if not self.nodes[i]:
                self.dfs(i)

    def is_connected(self):
        """ Check if graph is connected """
        seed_node = random.choice(list(self.nodes.keys()))
        self.dfs(seed_node)
        visited_nodes = list(self.nodes.values()).count(True)
        if len(self.nodes) == visited_nodes:
            return True
        else:
            return False


def check_leaf_nodes(graph):
    """
    Employs a simple and cheap heuristic to count leaf nodes
    """

    subjects = list();
    objects = list()

    for (s, p, o) in graph:
        subjects.append(s);
        objects.append(o)

    leaf_nodes = list(set(subjects) - set(objects)) + list(set(objects) - set(subjects))
    return leaf_nodes


def follow_direction(graph, start_node):
    """ Checks if path graph follows edges """

    edge_connections = dict()
    for triple in graph:
        (source, _, target) = triple
        # A node should not be a source more than once
        if source in edge_connections:
            return False
        edge_connections[source] = target

    i = start_node
    for _ in range(len(edge_connections)):
        if i in edge_connections:
            f = edge_connections[i]
            edge_connections.pop(i)
            # print(f'{i} -> {f}', 'remaining:', len(edge_connections))
            i = f
        else:
            return False

    if len(edge_connections) != 0:
        return False

    return True


def check_pathgraph(graph, entity_labels, relation_labels, length=None):
    """
    Check for path graphs

    Any lengths of path graphs are allowed
    Acyclic graphs are not counted as paths.
    """

    # Check if the graph has the expected graph length
    # if length is not None and len(graph) != length:
    #     return False

    if not len(graph) > 1:
        return False

    # Check for null nodes
    for s, _, o in graph:
        if s == len(entity_labels) or o == len(entity_labels):
            return False

    # Label subject, object, predicate - Not required here
    # graph = [[entity_labels[int(s)], relation_labels[int(p)], entity_labels[int(o)]] for s, p, o in unlabelled_graph]

    # Acyclic pathgraph should have two leaf nodes
    leaf_nodes = check_leaf_nodes(graph)
    # print('leaf_nodes', leaf_nodes)
    if len(leaf_nodes) != 2:
        return False

    # Check path graph follows the edge directions from either one of the leaf nodes
    if not (follow_direction(graph, start_node=leaf_nodes[0]) or
            follow_direction(graph, start_node=leaf_nodes[1])):
        return False

    return True


inductive_nodes = [
    '_academic',
    '_time',
]

# List of people
people = [
    'Eveleen Eszes',
    'Meera Mac Amhalghaidh',
    'Cleophas Erős',
    'Vanja Rutkowski',
    'Hildr Liang',
    'Hüseyn Mikkelsen',
    'Schwanhild Patel',
    'Wolodymyr Plamondon',
    'Martina Alan',
    'Nirmala Kwiatkowski',
    'Gríðr Oberst',
    'Aneirin Logan',
    'Pili Barbier',
    'Danya Maçon',
    'Vesta Corna',
    'Priscilla Stringer',
    'Rhianon Van Rompu'
    'Valerija Alger',
    'Renáta Park',
    'Anita McCabe',
    'Rahul Rowan',
    'Lulu MacKay',
    'Aeson Pinheiro',
    'Toshe Pander',
    'Salvator Cousineau',
    'Tahlako Plaskett',
    'Teófilo Gupta',
    'Yonit Stojanovski',
    'Terminus Ó Coigligh',
    'Erna Van Dijk',
    'Hildegard Dávid',
    'Hilda Milošević',
    'Bogdan Baglio',
    'Eldar Dawson',
    'Anaïs Hajós',
    'Boglárka Rosenberg',
    'Božidar Bullard',
    'Anselm Sulzbach',
    'Toygar Pesce',
    'Nikita Serafim',
    'Diindiisi Fortuyn',
    'Amalia Görög',
    'Kunti Andréasson',
    'Romana Sitko',
    'Athanas Blackbug',
    'Marjan Albert',
    'Drusus Krejči',
    'Livnat Hull',
    'Casimir Darnell',
    'Vojtech Mac Neachtain',
    'Lída McDaniel'
]

# List of academic roles
roles = [
    'professor',
    'assistant professor',
    'post doctoral researcher',
    'phd researcher',
    'masters researcher',
]

# List of years
years = list(range(1950, 2023))

academic_graph_relations = [
    'has_time',
    'has_role',
    'has_name',
    'start_year',
    'end_year'
]


def check_academic_graph(graph, entity_labels, relation_labels, length=None):
    """ Checks the validity of a graph """
    v = dict()

    # Check if the graph has the expected graph length
    # if length is not None and len(unlabelled_graph) != length:
    #     return False

    if not len(graph) > 1:
        return False

    # Check for null nodes
    for s, _, o in graph:
        if s == len(entity_labels) or o == len(entity_labels):
            return False

    # Label subject, object, predicate
    # graph = [[entity_labels[int(s)], relation_labels[int(p)], entity_labels[int(o)]] for s, p, o in graph]

    for triple in graph:
        s, p, o = triple

        # If a subject and predicate exist more than once, then it is not valid
        if (s, p) in v:
            return False

        v[(s, p)] = o

        # Check if the objects have the valid object type
        if not is_valid_triple(s, p, o):
            return False

    # Check graph pattern
    if not check_graph_pattern(v):
        return False

    # Temporal reasoning - Check if end year is higher than start year
    if not v[('_time', 'end_year')] >= v[('_time', 'start_year')]:
        return False

    return True


def check_graph_pattern(graph):
    """ Checks if a graph follows the graph pattern """

    if not ('_academic', 'has_name') in graph:
        return False
    if not graph[('_academic', 'has_name')] in people:
        return False

    if not ('_academic', 'has_role') in graph:
        return False
    if not graph[('_academic', 'has_role')] in roles:
        return False

    if not ('_academic', 'has_time') in graph:
        return False
    if not graph[('_academic', 'has_time')] == '_time':
        return False

    if not ('_time', 'start_year') in graph:
        return False
    if not graph[('_time', 'start_year')] in years:
        return False

    if not ('_time', 'end_year') in graph:
        return False
    if not graph[('_time', 'end_year')] in years:
        return False

    return True


def is_valid_triple(s, p, o):
    """ Check if a triple is semantically valid """
    if p == 'has_role' and o not in roles:
        return False

    if p == 'has_name' and o not in people:
        return False

    if p == 'start_year' and o not in years:
        return False

    if p == 'end_year' and o not in years:
        return False

    return True


# List of countries
countries = [
    'Serbia',
    'United Kingdom',
    'Ireland',
    'Greece',
    'Ukraine',
    'Spain',
    'Switzerland',
    'Norway',
    'Iceland',
    'Sweden'
]

# List of languages
languages = [
    'German',
    'Irish',
    'Russian',
    'French',
    'Czech',
    'Italian',
    'Norwegian',
    'Greek',
    'Dutch',
    'English'
]

# List of cities
cities = [
    'Warsaw',
    'Lisbon',
    'Bern',
    'Dublin',
    'Amsterdam',
    'London',
    'Paris',
    'Madrid',
    'Budapest',
    'Copenhagen'
]


def is_valid_location_triple(triple):
    """ Check if a triple is semantically valid """

    s, p, o = triple

    # Type constraint 1: Language spoken_in Country
    if p == 'spoken_in':
        if s in languages and o in countries:
            return True
        else:
            return False

    # Type constraint 2: City part_of Country
    if p == 'part_of':
        if s in cities and o in countries:
            return True
        else:
            return False

    # Type constraint 3: Same_as relations
    if p == 'same_as':
        # Language same_as Language
        if s in languages and o in languages:
            return True

        # City same_as City
        if s in cities and o in cities:
            return True

        # City same_as City
        if s in countries and p == 'same_as' and o in countries:
            return True

    return False


def check_location_graph(graph, entity_labels, relation_labels, length=None):
    """ Check the semantic validity of graphs """

    # # Check if the graph has the expected graph length
    # if length is not None and len(unlabelled_graph) != length:
    #     return False
    if not len(graph) > 1:
        return False

    # Check for null nodes
    for s, _, o in graph:
        if s == len(entity_labels) or o == len(entity_labels):
            return False

    # Label subject, object, predicate
    # graph = [[entity_labels[int(s)], relation_labels[int(p)], entity_labels[int(o)]] for s, p, o in graph]

    # Check if all triples have valid entity types
    for triple in graph:
        if not is_valid_location_triple(triple):
            return False

    return True


def check_movie_graph(graph, entity_labels, relation_labels, length=None):
    """
    Verify whether a particular graph satisfies that stated semantics. For this dataset, that means:

    - This inductive node does not connect to itself by any relation
    - There is at least one person connected to the inductive node by the director relation
    - There is at least one person connected to the inductive node by the actor relation.
    - Only the inductive node occurs in the subject position of any triples, and the inductive
      node only ever occurs in the subject position of any triples.

    NB: This method assumes that `_movie` is the label of the inductive node and that any other nodes have valid
    transductive labels. This is not checked.

    :param graph: The graph, represented as an iterable of string triples.

    ":returns: True is the graph satisfies the semantics, false if it doesn't
    """

    # Check for null nodes
    for s, _, o in graph:
        if s == len(entity_labels) or o == len(entity_labels):
            return False

    # Label subject, object, predicate
    # graph = [[entity_labels[int(s)], relation_labels[int(p)], entity_labels[int(o)]] for s, p, o in graph]

    diredges = 0
    actedges = 0
    for s, p, o in graph:
        if s != '_movie':
            return False

        if o == '_movie':
            return False

        if p == 'has_director':
            diredges += 1

        if p == 'has_actor':
            actedges += 1

    if diredges < 1 or actedges < 1:
        return False
    return True


def fail(use_assert):
    if use_assert:
        assert False
    return False

def ordinal(i):
    """
    Label for the i-th ordinal. Note that this is not an inductive node: it represents the same thing in multiple graphs.

    :param i:
    :return:
    """
    return f'ordinal_{i:03}'


def check_article_graph(graph, entity_labels, relation_labels, length=None, ua=False):
    """
    Verify whether a particular graph satisfies that stated semantics. For this dataset, that means:

    - There is one or more triple with the relation `has_author`.
        - Exactly one node is the subject of all of these. Call this the article node.
        - The article node is labeled '_article' or labeled with an IRI
        - The object of every has_author triple has a label starting with '_authorpos'
        - Every _authorpos node is the object of only this triple.
        - Every _authorpos node is the subject of exactly two triples:
            - One with the relation `has_name`. The object of this triple is an IRI or starts with `_author`
            - One with the relation `has_order`. The object of this triple starts with `ordinal_`
        - If there are n authorpo nodes, then taken together, all their ordinals coincide with the range from one to n
          inclusive.

    - There are zero or more triples with the relation `has_reference`.
        - The subject of all such triples is the article node
        - The object of all such triples is an IRI

    - There are zero or more triples with the relation `has_subject`
        - The object of all such triples is the article node.
        - The subject of all such triples starts with `_subject` or is an IRI

    - There are zero or more triples with the relation `subclass_of`
        - The object and subject of such a triple either start with `_subject` or are IRIs
        - Either the subject of the triple is connected to the article by a `has_subject` relation or it is connected to
          such a subject by a chain of `subclass_of` relations

    Note that types of transductive nodes are not enforced. A graph is valid if it uses an author node as a subject or
    as a reference. For this dataset, these are considered "soft" constraints. Learning these well should be reflected
    in the bits-per-graph evaluation metric.

    :param graph: The graph, represented as an iterable of string triples.

    :returns: True if the graph satisfies the semantics, false if it doesn't
    """

    PREDICATES = ['has_author', 'has_name', 'has_order', 'cites', 'has_subject', 'subclass_of']

    s2t = {}
    p2t = {}
    o2t = {}

    for p in PREDICATES:
        p2t[p] = []

    for s, p, o in graph:

        if s not in s2t:
            s2t[s] = []

        if o not in o2t:
            o2t[o] = []

        if p not in p2t:

            strgraph = ''
            for t in graph:
                strgraph += str(t) + '\n'

            # assert False, f'{p}\n\n {strgraph}'
            # -- This is not an invalid graph, it's a bug.
            return False

        s2t[s].append((s, p, o))
        p2t[p].append((s, p, o))
        o2t[o].append((s, p, o))

    # - There is one or more triple with the relation `has_author`.
    if len(p2t['has_author']) < 1:
        return fail(ua)

    # -- Exactly one node is the subject of all of these. Call this the article node.
    subjects = {s for s, p, o in p2t['has_author']}
    if len(subjects) != 1:
        return fail(ua)

    article = next(iter(subjects))

    # -- The article node is labeled '_article' or labeled with an IRI
    if not (article == '_article' or article.startswith('http')):
        return fail(ua)

    # -- The object of every has_author triple has a label starting with '_authorpos'
    ordinals = []
    num_authors = 0
    for s, p, o in p2t['has_author']:

        num_authors += 1

        if not o.startswith('_authorpos'):
            return fail(ua)

        # -- Every _authorpos node is the object of only this triple.
        if not (len(o2t[o]) == 1):
            return fail(ua)

        if o not in s2t:  # Thiviyan
            return fail(ua)

        # -- Every _authorpos node is the subject of exactly two triples:
        if not len(s2t[o]) == 2:
            return fail(ua)

        # --- One with the relation `has_name`. The object of this triple is an IRI or starts with `_author`
        # --- One with the relation `has_order`. The object of this triple starts with `ordinal_`
        rels = {p for _, p, _ in s2t[o]}
        if not rels == {'has_name', 'has_order'}:
            return fail(ua)

        for s, p, o in s2t[o]:
            if p == 'has_name':
                if not (o.startswith('_author') or o.startswith('http')):
                    return fail(ua)
            elif p == 'has_order':
                if not o.startswith('ordinal_'):
                    return fail(ua)

                ordinals.append(o)
            # else:
            #     assert False, p

    # -- If there are n authorpos nodes, then taken together, all their ordinals coincide with the range from one to n
    #    inclusive.
    ord_target = [ordinal(i) for i in range(1, num_authors + 1)]

    if not (sorted(ord_target) == sorted(ordinals)):
        return fail(ua)

    # - There are zero or more triples with the relation `has_reference`.
    for s, p, o in p2t['cites']:
        # -- The subject of all such triples is the article node
        if s != article:
            return fail(ua)

        # -- The object of all such triples is an IRI
        if not o.startswith('http'):
            return fail(ua)

    # - There are zero or more triples with the relation `has_subject`
    for s, p, o in p2t['has_subject']:
        # -- The object of all such triples is the article node.
        if s != article:
            return fail(ua)
        # -- The subject of all such triples starts with `_subject` or is an IRI
        if not (o.startswith('http') or o.startswith('_subject')):
            return fail(ua)

    # - There are zero or more triples with the relation `subclass_of`
    for s, p, o in p2t['subclass_of']:
        # -- The object and subject of such a triple either start with `_subject` or are IRIs
        if not (s.startswith('http') or s.startswith('_subject')):
            return fail(ua)
        if not (o.startswith('http') or o.startswith('_subject')):
            return fail(ua)

    # -- Either the subject of the triple is connected to the article by a `has_subject` relation or it is connected to
    #    such a subject by a chain of `subclass_of` relations
    subjects = {o for _, _, o in p2t['has_subject']}
    scpairs = [(s, o) for s, p, o in p2t['subclass_of']]

    changed = True
    while changed:

        nwscpairs = []

        for s, o in scpairs:
            if s in subjects:
                subjects.add(o)
            else:
                nwscpairs.append((s, o))

        changed = len(nwscpairs) != len(scpairs)

        scpairs = nwscpairs

    if not len(scpairs) == 0:
        return fail(ua)

    return True
