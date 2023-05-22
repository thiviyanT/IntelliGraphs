<p  align="center">
    <img src="images/IntelliGraph-logo.png" width="450px;" style="max-width: 100%;  margin-right:10px;">
    <h3 align="center">
        Benchmark Datasets for Knowledge Graph Generation
    </h3>
<p>

<p align=center>
    <a href="https://pypi.org/project/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=PyPI%20package&logo=pypi&logoColor=white" title="The current version of IntelliGraphs"></a>
    <a href="https://anaconda.org/thiv/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=Anaconda.org&logo=anaconda&logoColor=white" title="IntelliGraphs on Conda"></a>
    <a href="https://github.com/intelligraphs/layout-parser/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/intelligraphs" title="IntelliGraphs uses MIT License"></a>
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/intelligraphs">
</p>

<p align=center>
    <a href="https://doi.org/10.5281/zenodo.7824818"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7824818.svg" title="DOI of IntelliGraphs"></a>
<!---
    <a href="https://arxiv.org"><img src="https://img.shields.io/badge/paper-2103.15348-b31b1b.svg" title="IntelliGraphs Paper"></a>
    <a href="https://intelligraphs.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/doc-layout--parser.readthedocs.io-light.svg" title="IntelliGraphs Documentation"></a>
--->
</p>

---

IntelliGraphs is a Python package that generates a collection of benchmark datasets. These datasets are intended to be used
for benchmarking machine learning models under transductive settings. It can also be used as a testbed for developing
new generative models. This library was designed to be extendable to create new synthetic datasets with custom 
First-Order Logical (FOL) rules.

### TODO

* Ask Paul: Do we want to register this with Zenodo? If so, add DOI badge here.
* Ask Paul: How to generate dataset metadata? Is it needed?
* Ask Paul: When to register dataset? Before or after publication?
* Ask Paul: Where to put the dataset so that it lasts? (Zenodo, GitHub, etc.)
* Ask Peter/Paul: Do we want to make it available on PyPI? If so, add badge here.
* Make GitHub repo anonymous before submission

## Table of Contents

* [Installation](#installation)
* [Advantages](#advantages)
* [Usage](#usage)
* [IntelliGraphs Data Loader](#intelligraphs-data-loader)
* [Datasets](#datasets)
* [Citation](#citation)
* [License](#license)
* [Contributing](#contributing)
* [Acknowledgements](#acknowledgements)


## Installation

To install IntelliGraphs, using pip:

```bash
pip install intelligraphs
```

Or using conda:

```bash
conda install -c thiv intelligraphs
```

## Advantages

* **Easy to use**: Generate and manipulate Knowledge Graphs with a simple and clean Python API.
* **Flexible**: Customize the number of graphs, triples, and data splits.
* **Extendable**: Create more graphs according to custom FOL rules.
* **Efficient**: Fast and memory-efficient graph generation and manipulation using native Python data structures.
* **Visualization**: Visualize Knowledge Graphs.

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

Statements in FOL are made up of two parts: the subject and the predicate. The subject is the thing that is being
described, and the predicate is the property of the subject. For example, the statement "John is a student" has the
subject "John" and the predicate "is a student".

FOL statements can be expressed in a text file. For examples, the FOL statements for the syn-paths can be expressed as:
```text
Connected(x) → (Subject(x, y) ∨ Object(x, z))
∀x ∀y (¬Root(x) ∨ ¬Object(y, x))
∀x ∀y (¬Leaf(x) ∨ ¬Subject(y, x))
```

This can be parsed by the IntelliGraphs library using:

```python
intelligraph.parse_fol_rules('path/to/rules.txt')
```

## Reproducibility

The datasets will be uploaded to Zenodo. We will update this README with the DOI once it is available.

Changes to the datasets will be documented in the [CHANGELOG](CHANGELOG.md), and the version numbers will follow [Semantic Versioning](https://semver.org/).

## Reporting Issues

If you encounter any bugs or have any feature requests, please file an issue [here](https://github.com/thiviyanT/IntelliGraphs/issues).

## How to Cite

If you use IntelliGraphs in your research, please cite the following paper:

```bibtex
[INSERT BIBTEX OF ARXIV PREPRINT HERE]
```

## Future Work

* **Inductive Setting** It would be very useful doing the data split such that it allows for inductive setting.

* **2D and 3D Scene Graph Generation with corresponding knowledge graphs** Here are some examples of 2D and 3D scene graphs:

<p  align="center">
    <img src="images/2d_objects.jpg" width="200px;" style="max-width: 100%;  margin-right:10px;">
    <img src="images/3d_objects.jpg" width="400px;" style="max-width: 100%;  margin-right:10px;">
<p>


## License

IntelliGraphs is licensed under MIT License. See [LICENSE](LICENSE) for more information.

Copyright (c) 2023 Thiviyan Thanapalasingam
