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
    <a href="https://arxiv.org/abs/2307.06698"><img src="https://img.shields.io/badge/preprint-2307.06698-b31b1b.svg" title="IntelliGraphs Paper"></a>
<!---
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
* [Reporting Issues](#reporting-issues)
* [License](#license)


## About IntelliGraphs

IntelliGraphs is a Python package that generates a collection of benchmark datasets. These datasets are intended to be used
for benchmarking machine learning models under transductive settings. It can also be used as a testbed for developing
new generative models. This library was designed to be extendable to create new synthetic datasets with custom 
First-order logical (FOL) rules.

### Advantages

* **Easy to use**: Generate and manipulate Knowledge Graphs with a simple and clean Python API.
* **Flexible**: Customize the number of graphs, triples, and data splits.
* **Extendable**: Create more graphs according to custom FOL rules.
* **Efficient**: Fast and memory-efficient graph generation and manipulation using native Python data structures.
* **Visualization**: Visualize Knowledge Graphs.


## Installation

To install IntelliGraphs, use pip:

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

The datasets required for this project can be obtained either manually or automatically through IntelliGraphs.

#### Manual Download

The datasets are hosted on Zenodo and can be downloaded directly from the following link:

**Zenodo Download Link:** [https://doi.org/10.5281/zenodo.7824818](https://doi.org/10.5281/zenodo.7824818)

To manually download the datasets:

1. Click on the provided link above.
2. You will be redirected to the Zenodo page hosting the datasets.
3. On the Zenodo page, click the **Download** button or select specific files to download as needed.
4. Once downloaded, extract the files (if compressed) to a directory of your choice on your local machine.

#### Automatic Download with IntelliGraphs

Alternatively, you can use the IntelliGraphs tool to download and prepare the datasets automatically. This method is convenient if you want to streamline the process and ensure that all required data is correctly organized.

To download datasets automatically:

1. Ensure you have IntelliGraphs installed and properly configured. If not, refer to the IntelliGraphs documentation for setup instructions.
2. Use the IntelliGraphs download command or script provided in the project (usually a script named `download_datasets.py` or similar).
3. Run the script, and it will automatically download the datasets from Zenodo, extract them, and place them in the appropriate directories for use.

It will download the specified dataset files into a `.data` directory. 

## Reporting Issues

If you encounter any bugs or have any feature requests, please file an issue [here](https://github.com/thiviyanT/IntelliGraphs/issues).

## How to Cite

If you use IntelliGraphs in your research, please cite the following paper:

```bibtex
@article{thanapalasingam2023intelligraphs,
  title={IntelliGraphs: Datasets for Benchmarking Knowledge Graph Generation},
  author={Thanapalasingam, Thiviyan and van Krieken, Emile and Bloem, Peter and Groth, Paul},
  journal={arXiv preprint arXiv:2307.06698},
  year={2023}
}
```

## License

IntelliGraphs is licensed under CC-BY License. See [LICENSE](LICENSE) for more information.
