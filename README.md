<p  align="center">
    <img src="images/IntelliGraph-logo.png" width="450px;" style="max-width: 100%;  margin-right:10px;">
    <h3 align="center" >
        Benchmark Datasets for Knowledge Graph Generation
    </h3>
<p>

<p align=center>
    <a href="https://pypi.org/project/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=PyPI%20package&logo=pypi&logoColor=white" title="The current version of IntelliGraphs"></a>
    <a href="https://anaconda.org/thiv/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=Anaconda.org&logo=anaconda&logoColor=white" title="IntelliGraphs on Conda"></a>
    <a href="https://github.com/intelligraphs/layout-parser/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/intelligraphs" title="IntelliGraphs uses CC-BY License"></a>
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/intelligraphs">
</p>

<p align=center>
    <a href="https://doi.org/10.5281/zenodo.7824818"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7824818.svg" title="DOI of IntelliGraphs"></a>
<!---
    <a href="https://arxiv.org"><img src="https://img.shields.io/badge/paper-2103.15348-b31b1b.svg" title="IntelliGraphs Paper"></a>
    <a href="https://intelligraphs.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/doc-layout--parser.readthedocs.io-light.svg" title="IntelliGraphs Documentation"></a>
--->
</p>

<!---
TODO: 
* Make sure that the semantic checker is doing FOL as written down below 
* Mention in natural language that the size constraint applies to synthetic graphs. Although, it can be expressed in first order logic we leave it out for brevity.
* Model checking complexity increases with the number of rules
* Check if the reasoner is strong enough - SAT solving
--->

## Table of Contents

* [About IntelliGraphs](#about-intelligraphs)
* [Installation](#installation)
* [Advantages](#advantages)
* [Usage](#usage)
* [IntelliGraphs Data Loader](#intelligraphs-data-loader)
* [Datasets](#datasets)
* [First-Order Logic](#first-order-logic)
* [Baseline Models](#baseline-models)
* [Reporting Issues](#reporting-issues)
* [How to Contribute](#how-to-contribute)
* [License](#license)


## About IntelliGraphs

IntelliGraphs is a collection of datasets for benchmarking Knowledge Graph Generation models. 
This is a Python package that loads the datasets and verifing graphs using logical rules expressed in 
First-Order Logical (FOL) rules. It can also be used as a testbed for developing
new generative models.  

### Long-term Storage of Datasets

The datasets have been uploaded to Zenodo. Here is the link: https://doi.org/10.5281/zenodo.7824818

### Advantages

* **Easy to use**: Generate and manipulate Knowledge Graphs with a simple and clean Python API.
* **Flexible**: Customize the number of graphs, triples, and data splits.
* **Extendable**: Create more graphs according to custom FOL rules.
* **Efficient**: Fast and memory-efficient graph generation and manipulation using native Python data structures.
* **Visualization**: Visualize Knowledge Graphs.


## Installation

To install IntelliGraphs, using pip:

```bash
pip install intelligraphs
```

Or using conda:

```bash
conda install -c thiv intelligraphs
```

## Usage

Here's a brief example of how to use various features of the IntelliGraphs library:

```python
from intelligraphs import IntelliGraphs

# Create an instance of IntelliGraphs with 10 graphs, variable length triples, and a random seed of 42
intelligraph = IntelliGraphs(random_seed=42, num_graphs=10, var_length=True, min_triples=2, max_triples=5)

# Manually generate the graphs
intelligraph.generate_graphs()

# Get the list of graphs
graphs = intelligraph.get_graphs()

# Print the first graph
intelligraph.print_graph(graphs[0])

# Visualize the first graph
intelligraph.visualize_graph(graphs[0])

# Get the natural language sentences for the triples
all_sentences = intelligraph.to_natural_language()

# Print the sentences for each graph
for i, sentences in enumerate(all_sentences):
    print(f"Graph {i + 1}:")
    for sentence in sentences:
        print(sentence)
    print()

# Manually trigger splitting the data into train, valid, and test sets
intelligraph.split_data(split_ratio=(0.6, 0.3, 0.1))

# Get the data splits
splits = intelligraph.get_splits()

# Print the data splits
for split_name, data in splits.items():
    print(f"{split_name.capitalize()} Data:")
    for graph in data:
        print(graph)
    print()

# Save the graphs and splits to text files
intelligraph.save_graphs(filename='example', file_path='output', zip_compression=False)
intelligraph.save_splits(filename='example', file_path='output', zip_compression=False)

# Save the graphs and splits to zip compressed text files
intelligraph.save_graphs(filename='example', file_path='output', zip_compression=True)
intelligraph.save_splits(filename='example', file_path='output', zip_compression=True)
```

## IntelliGraphs Data Loader

The `IntelliGraphsDataLoader` class is a utility for loading IntelliGraphs datasets, simplifying the process of accessing and organizing the data for machine learning tasks. It provides functionalities to download, extract, and load the datasets into PyTorch tensors.

### Features and Benefits
* **PyTorch Integration**: Seamless integration with PyTorch for graph-based machine learning tasks.
* **Configuration Options**: Customizable options for batch size, padding, and shuffling.
* **Reproducibility**: Facilitates documentation and reproduction of experiments for improved research integrity.
* **Data Preprocessing**: Simplifies dataset preprocessing with automated download, extraction, and conversion.

### Usage
1. Instantiate the DataLoader:
``` python
from intelligraphs import IntelliGraphsDataLoader
data_loader = IntelliGraphsDataLoader(dataset_name='syn-paths')
```
2. Load the Data:
``` python
train_loader, valid_loader, test_loader = data_loader.load_torch(
    batch_size=32,
    padding=True,
    shuffle_train=False,
    shuffle_valid=False,
    shuffle_test=False
)
```
3. Access the Data:
``` python
for batch in train_loader:
    # Perform training steps with the batch

for batch in valid_loader:
    # Perform validation steps with the batch

for batch in test_loader:
    # Perform testing steps with the batch
```

## Datasets

### Downloading the Datasets

The datasets can be downloaded manually or automatically without IntelliGraphs.

#### Manual Download

The datasets can be directly downloaded from the following links:
* syn-paths.zip: https://www.dropbox.com/s/kp1xp2rbieib4gl/syn-paths.zip?dl=1
* syn-tipr.zip: https://www.dropbox.com/s/wgm2yr7h8dhcj52/syn-tipr.zip?dl=1
* syn-types.zip: https://www.dropbox.com/s/yx7vrvsxme53xce/syn-types.zip?dl=1
* wd-articles.zip: https://www.dropbox.com/s/37etzy2pkix84o8/wd-articles.zip?dl=1
* wd-movies.zip: https://www.dropbox.com/s/gavyilqy1kb750f/wd-movies.zip?dl=1


#### Automatic Download

You can also download and unzip the datasets using the download.sh script. This 
script will download all five datasets: 

```bash
bash download.sh
```

Wait for the script to complete. It will download the specified zip files from the provided URLs and unzip them into a 
`.data` directory. Once the script finishes executing, you can find the extracted files in the `.data` directory.

### Dataset Statistics

| Dataset | Rules | # Nodes | # Edges | # Relations | # Classes | # Train | # Valid | # Test |
|---------|-------------|---------|---------|-------------|-----------|---------|---------|--------|
|syn-paths|-|-|-|-| - |-|-|-|
|syn-tipr|-|-|-|-|-|-|-|-|
|syn-types|-|-|-|-|-|-|-|-|
|wd-movies|-|-|-|-|-|-|-|-|
|wd-articles|-|-|-|-|-|-|-|-|

### Example

<table>
  <tr>
    <th>Dataset</th>
    <th>Knowledge Graph</th>
  </tr>
  <tr>
    <td>syn-paths</td>
    <td><pre>
    </pre></td>
  </tr>
  <tr>
    <td>syn-tipr</td>
    <td><pre>
    </pre></td>
  </tr>
  <tr>
    <td>syn-types</td>
    <td><pre>
    </pre></td>
  </tr>
  <tr>
    <td>wd-movies</td>
    <td><pre>
    </pre></td>
  </tr>
  <tr>
    <td>wd-articles</td>
    <td><pre>
    </pre></td>
  </tr>
</table>
    
    
## First-Order Logic

First-order logic (FOL) is a logic system that is used to describe the world around us. It is a formal language that
allows us to make statements about the world.

Intelligraphs can process FOL statements that are expressed in `.txt` files. This can be parsed by the IntelliGraphs library using:

```python
intelligraph.parse_fol_rules('path/to/rules.txt')
```

### SYN-PATHS

#### Natural language: 

* It should be a connected graph, i.e. each subject and object should be connected to at least one other triple.
* Must contain directed edges
* directions of the edges should follow a path from the root to the leaf nodes

#### FOL statements:
```text
forall x,y,z connected(x,y) ^ connected(y,z) -> connected(x,z)
forall x,y edge(x,y) -> connected(x,y)

exists x root(x)
forall x,y root(x) ^ root(b) -> a=b
forall x root(x) <-> forall y ¬ edge(y,x)

forall x,y connected(x,y) -> x≠y
forall x root(x) -> forall y connected(x,y) v x=y
forall x,y,z edge(y,x) ^ edge(z,x) -> y=z
forall x,y,z edge(x,y) ^ edge(x,z) -> y=z

forall x,y edge(x,y) <-> cycle_to(x,y) v drive_to(x,y) v train_to(x,y)
```

### SYN-TYPES:

#### Natural language: 

Must satisfy the following type constraints:
* the predicate  'spoken in' can only exist between a 'language' and 'country'
* the predicate 'part of' can only exist between a city and a country.
* the predicate 'same as' can only exist between the same types (language, city, country)

#### FOL statements:
```text
forall x,y spoken_in(x, y) -> (language(x) ^ country(y))
forall x,y part_of(x, y) -> (city(x) ^ country(y)))
forall x,y (same_as(x, y) -> (language(x) ^ language(y)) v (city(x) ^ city(y)) v (country(x) ^ country(y))

forall x language(x) -> ¬ country(x) ^ ¬ city(x)
forall x country(x) -> ¬ language(x) ^ ¬ city(x)
forall x city(x) -> ¬ language(x) ^ ¬ country(x)

forall x,y same_as(x, y) -> ¬ same_as(x, y)
```

### SYN-TIPR:

#### Natural language: 

Must contain all edges:
• has role( academic, role) ∧ has name( academic, name)∧
has time( academic, time) ∧ start year( time, year) ∧ end year( time, year)
• end year ≥ start year

#### FOL statements:
```text
forall x,y has_role(x, y) -> academic(x) ^ role(y)
forall x,y has_name(x, y) -> academic(x) ^ name(y)
forall x,y has_time(x, y) -> academic(x) ^ time(y)
forall x,y start_year(x, y) -> time(x) ^ year(y)
forall x,y end_year(x, y) -> time(x) ^ year(y)
forall x,y,z end_year(x, y) ^ start_year(x, z) -> after(y, z)

forall x ¬ has_role(x, x)
forall x ¬ has_name(x, x)
forall x ¬ has_time(x, x)
forall x ¬ start_year(x, x)
forall x ¬ end_year(x, x)

forall x academic(x) -> ¬ role(x) ^ ¬ time(x) ^ ¬ name (x) ^ ¬ year(x)
forall x role(x) -> ¬ academic(x) ^ ¬ time(x) ^ ¬ name (x) ^ ¬ year(x)
forall x time(x) -> ¬ academic(x) ^ ¬ role(x) ^ ¬ name (x) ^ ¬ year(x)
forall x year(x) -> ¬ academic(x) ^ ¬ role(x) ^ ¬ name (x) ^ ¬ time(x)
forall x name(x) -> ¬ academic(x) ^ ¬ role(x) ^ ¬ year (x) ^ ¬ time(x)
```

### WD-MOVIES:

#### Natural language:
* This inductive node does not connect to itself by any relation
* There is at least one person connected to the inductive node by the has_director relation 
* There is at least one person connected to the inductive node by the has_actor relation. 
* There is at least one genre connected to the inductive node by the has_genre relation. 
* Only the inductive node occurs in the subject position of any triples, and the inductive node only ever occurs in the subject position of any triples.

NB: This method assumes that `_movie` is the label of the inductive node and that any other nodes have valid transductive labels. This is not checked.

Question: Sets of directors and actors: Mutually exclusive? 

#### FOL statements:

Assume we have constants **inductive_nodes**, which means ...
```text
forall x,y connected(x,y) <-> has_director(x,y) v has_actor(x,y) v has_genre(x,y)

exists x has_director(x, inductive_node)
exists x has_actor(x, inductive_node)
exists x has_genre(x, inductive_node)

forall x x ≠ inductive_node -> connected(inductive_node, x) 
forall x,y x ≠ inductive_node ^ y ≠ inductive_node -> ¬connected(x, y)
forall x ¬connected(x, inductive_node)

forall x,y has_director(x,y) v has_actor(x,y) -> person(y)  (TODO: revisit this rule in the semantic checker)
forall x,y has_genre(x,y) -> genre(y)  (TODO: revisit this rule in the semantic checker)
```

### WD-ARTICLES:

#### Natural language:
- There is one or more triple with the relation `has_author`.
  - Exactly one node is the subject of all of these. Call this the article node.
  - The article node is labeled '_article' or labeled with an IRI.  ***- Ask Peter if mutually exclusive **
  - The object of every has_author triple has a label starting with '_authorpos'
  - Every _authorpos node is the object of only this triple.
  - Every _authorpos node is the subject of exactly two triples:
       - One with the relation `has_name`. The object of this triple is an IRI or starts with `_author`
       - One with the relation `has_order`. The object of this triple starts with `ordinal_`
  - If there are n authorpos nodes, then taken together, all their ordinals coincide with the range from one to n
    inclusive.

- There are zero or more triples with the relation `cites`.
    - The subject of all such triples is the article node
    - The object of all such triples is an IRI

- There are zero or more triples with the relation `has_subject`
    - The object of all such triples is the article node.
    - The subject of all such triples starts with `_subject` or is an IRI

- There are zero or more triples with the relation `subclass_of`
    - The object and subject of such a triple either start with `_subject` or are IRIs
    - Either the subject of the triple is connected to the article by a `has_subject` relation or it is connected to
      such a subject by a chain of `subclass_of` relations

#### FOL statements:

```text

exists x has_author(article_node, x)

forall x,y connected(x,y) <-> has_author(x,y) v has_name(x,y) v has_order(x,y) v cites(x,y) v has_subject(x,y) v subclass_of(x,y)
forall x,y connected(x,y) -> ¬ connected(y,x) v cites(y, x)
forall x ¬ connected(x, x)

forall x, y has_author(x, y) -> x = article_node

article(article_node) v iri(article_node)

forall x has_author(article_node, x) -> authorpos(x)
forall x authorpos(x) <-> exists y has_order(x, y) ^ exists y has_name(x, y)
forall x,y has_order(x,y) -> authorpos(x) ^ ordinal(y)
forall x, y has_name(x, y) -> authorpos(x) ^ name(y) v iri(y)
forall x, y, z has_order(x, y) ^ has_order(x, z) -> y = z
forall x, y, z has_name(x, y) ^ has_order(x, z) -> y = z

forall x author(x) -> ¬ subject(x) ^ ¬ iri(x) ^ ¬ name(x) ^ ¬ oridinal(x) ^ ¬ author_pos(x) 
forall x subject(x) -> ¬ author(x) ^ ¬ iri(x) ^ ¬ name(x) ^ ¬ oridinal(x) ^ ¬ author_pos(x) 
forall x iri(x) -> ¬ author(x) ^ ¬ subject(x) ^ ¬ name(x) ^ ¬ oridinal(x) ^ ¬ author_pos(x) 
forall x name(x) -> ¬ subject(x) ^ ¬ iri(x) ^ ¬ author(x) ^ ¬ oridinal(x) ^ ¬ author_pos(x) 
forall x oridinal(x) -> ¬ subject(x) ^ ¬ iri(x) ^ ¬ name(x) ^ ¬ author(x) ^ ¬ author_pos(x) 
forall x author_pos(x) -> ¬ subject(x) ^ ¬ iri(x) ^ ¬ name(x) ^ ¬ author(x) ^ ¬ oridinal(x) 

forall x,y,z subclass2(x, y) ^ subclass2(y, z) -> subclass_of(x, z) 
forall x,y subclass_of(x,y) -> subclass2(x,y) ^ (iri(x) v subject(x)) ^ (iri(y) v subject(y))
forall x,y subclass_of(x,y) -> exists z subclass2(x,z) ^ has_subject(article_node, z)

forall x,y cites(x, y) -> iri(y) ^ x=article_node
forall x,y has_subject(x, y) -> (subject(y) v iri(y)) ^ x=article_node
```

## Baseline Models

Baseline Models for compressing graphs based on traditional Knowledge Graph Embedding methods: TransE, DistMult and ComplEx.

### Prerequisites

* Python 3.10
* PyTorch 1.12 ` pip install torch==1.12.0 `
* Weight & Biases ` pip install wandb `
* TQDM ` pip install tqdm `
* 
### Hyperparameters 

The hyperparameters for each experiment can be found under the `baselines/configs` folder.

### Experiments

The following commands can be used to train and test the baseline models:

```bash
python baselines/experiments/train_baseline.py  --config configs/syn-paths-complex.yaml
python baselines/experiments/train_baseline.py  --config configs/syn-paths-distmult.yaml
python baselines/experiments/train_baseline.py  --config configs/syn-paths-transe.yaml
```

```bash
python baselines/experiments/train_baseline.py  --config configs/syn-tipr-complex.yaml
python baselines/experiments/train_baseline.py  --config configs/syn-tipr-distmult.yaml
python baselines/experiments/train_baseline.py  --config configs/syn-tipr-transe.yaml
```

```bash
python baselines/experiments/train_baseline.py  --config configs/syn-types-complex.yaml
python baselines/experiments/train_baseline.py  --config configs/syn-types-distmult.yaml
python baselines/experiments/train_baseline.py  --config configs/syn-types-transe.yaml
```

```bash
python baselines/experiments/train_baseline.py  --config configs/wd-movies-complex.yaml
python baselines/experiments/train_baseline.py  --config configs/wd-movies-distmult.yaml
python baselines/experiments/train_baseline.py  --config configs/wd-movies-transe.yaml
```

```bash
python baselines/experiments/train_baseline.py  --config configs/wd-articles-complex.yaml
python baselines/experiments/train_baseline.py  --config configs/wd-articles-distmult.yaml
python baselines/experiments/train_baseline.py  --config configs/wd-articles-transe.yaml
```


## Reporting Issues

If you encounter any bugs or have any feature requests, please file an issue [here](https://github.com/thiviyanT/IntelliGraphs/issues).

## How to Cite

If you use IntelliGraphs in your research, please cite the following paper:

```bibtex
[ Submission under review. To be made available after acceptance.]
```

## License

IntelliGraphs is licensed under CC-BY License. See [LICENSE](LICENSE) for more information.

Copyright (c) 2023 Thiviyan Thanapalasingam
