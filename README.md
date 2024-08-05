<p  align="center">
    <img src="assets/IntelliGraphs-logo 222x222(32-bit).png" width="150px;" style="max-width: 100%;  margin-right:10px;">
    <h3 align="center" >
        Benchmark Datasets for Knowledge Graph Generation
    </h3>
<p>

<p align=center>
    <a href="https://pypi.org/project/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=PyPI%20package&logo=pypi&logoColor=white" title="The current version of IntelliGraphs"></a>
    <a href="https://anaconda.org/thiv/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=Anaconda.org&logo=anaconda&logoColor=white" title="IntelliGraphs on Conda"></a>
    <a href="https://github.com/intelligraphs/layout-parser/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC--BY-blue.svg" title="IntelliGraphs uses CC-BY License"></a>
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/intelligraphs">
</p>

<p align=center>
    <a href="https://doi.org/10.5281/zenodo.7824818"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7824818.svg" title="DOI of IntelliGraphs"></a>
    <a href="https://arxiv.org/abs/2307.06698"><img src="https://img.shields.io/badge/preprint-2307.06698-b31b1b.svg" title="IntelliGraphs Paper"></a>
<!---
    <a href="https://intelligraphs.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/doc-layout--parser.readthedocs.io-light.svg" title="IntelliGraphs Documentation"></a>
--->
</p>

IntelliGraphs is a collection of benchmark datasets specifically for use in 
benchmarking generative models for knowledge graphs. You can learn more about it in our preprint: 
[*IntelliGraphs: Datasets for Benchmarking Knowledge Graph Generation*](https://arxiv.org/abs/2307.06698).

<!---
TODO: 
* Make sure that the semantic checker is doing FOL as written down below 
* Mention in natural language that the size constraint applies to synthetic graphs. Although, it can be expressed in first order logic we leave it out for brevity.
* Model checking complexity increases with the number of rules
* Check if the reasoner is strong enough - SAT solving
--->

## Table of Contents

* [Installation](#installation)
* [Downloading IntelliGraphs datasets](#downloading-the-datasets)
* [IntelliGraphs Data Loader](#intelligraphs-data-loader)
* [IntelliGraphs KG Generator](#intelligraphs-synthetic-kg-generator)
* [IntelliGraphs Verifier](#intelligraphs-verifier)
* [Baseline Implementations](#baseline-implementations)
* [Reporting Issues](#reporting-issues)
* [License Information](#license-information)


## Installation

IntelliGraphs can be installed using either `pip` or `conda`, depending on your preferred package management system.

### Installing with pip

`pip` is the Python package installer, and it's commonly used for installing Python packages from the Python Package Index (PyPI). To install IntelliGraphs using `pip`, open your terminal or command prompt and run the following command:

```bash
pip install intelligraphs
```
This command will automatically download and install the IntelliGraphs package along with any dependencies required for it to function properly.

### Installing with conda

If you prefer to use conda, you can install IntelliGraphs by running the following command in your terminal or command prompt:

```bash
conda install -c thiv intelligraphs
```

### Verifying the Installation

After installation, you can verify that IntelliGraphs has been successfully installed by running the following command in your Python environment:

```python
import intelligraphs

print(intelligraphs.__version__)
```

### Platform Compatibility

This package has been tested and verified to work on macOS and Linux operating systems. 
However, users on Windows may encounter issues during installation or while running the package. 
If you experience any problems on Windows, please [raise an issue on the project's GitHub repository](https://github.com/thiviyanT/IntelliGraphs/issues).


[//]: # (## Getting Started)

[//]: # ()
[//]: # (Here's a brief example of how to use various features of the IntelliGraphs library:)

[//]: # ()
[//]: # (```python)

[//]: # (from intelligraphs import IntelliGraphs)

[//]: # ()
[//]: # (# Create an instance of IntelliGraphs with 10 graphs, variable length triples, and a random seed of 42)

[//]: # (intelligraph = IntelliGraphs&#40;random_seed=42, num_graphs=10, var_length=True, min_triples=2, max_triples=5&#41;)

[//]: # ()
[//]: # (# Manually generate the graphs)

[//]: # (intelligraph.generate_graphs&#40;&#41;)

[//]: # ()
[//]: # (# Get the list of graphs)

[//]: # (graphs = intelligraph.get_graphs&#40;&#41;)

[//]: # ()
[//]: # (# Print the first graph)

[//]: # (intelligraph.print_graph&#40;graphs[0]&#41;)

[//]: # ()
[//]: # (# Visualize the first graph)

[//]: # (intelligraph.visualize_graph&#40;graphs[0]&#41;)

[//]: # ()
[//]: # (# Get the natural language sentences for the triples)

[//]: # (all_sentences = intelligraph.to_natural_language&#40;&#41;)

[//]: # ()
[//]: # (# Print the sentences for each graph)

[//]: # (for i, sentences in enumerate&#40;all_sentences&#41;:)

[//]: # (    print&#40;f"Graph {i + 1}:"&#41;)

[//]: # (    for sentence in sentences:)

[//]: # (        print&#40;sentence&#41;)

[//]: # (    print&#40;&#41;)

[//]: # ()
[//]: # (# Manually trigger splitting the data into train, valid, and test sets)

[//]: # (intelligraph.split_data&#40;split_ratio=&#40;0.6, 0.3, 0.1&#41;&#41;)

[//]: # ()
[//]: # (# Get the data splits)

[//]: # (splits = intelligraph.get_splits&#40;&#41;)

[//]: # ()
[//]: # (# Print the data splits)

[//]: # (for split_name, data in splits.items&#40;&#41;:)

[//]: # (    print&#40;f"{split_name.capitalize&#40;&#41;} Data:"&#41;)

[//]: # (    for graph in data:)

[//]: # (        print&#40;graph&#41;)

[//]: # (    print&#40;&#41;)

[//]: # ()
[//]: # (# Save the graphs and splits to text files)

[//]: # (intelligraph.save_graphs&#40;filename='example', file_path='output', zip_compression=False&#41;)

[//]: # (intelligraph.save_splits&#40;filename='example', file_path='output', zip_compression=False&#41;)

[//]: # ()
[//]: # (# Save the graphs and splits to zip compressed text files)

[//]: # (intelligraph.save_graphs&#40;filename='example', file_path='output', zip_compression=True&#41;)

[//]: # (intelligraph.save_splits&#40;filename='example', file_path='output', zip_compression=True&#41;)

[//]: # (```)

## Downloading the Datasets

The datasets required for this project can be obtained either manually or automatically through IntelliGraphs Python package.

### Manual Download

The datasets are hosted on Zenodo and can be downloaded directly from the following link:

**Zenodo Download Link:** [https://doi.org/10.5281/zenodo.7824818](https://doi.org/10.5281/zenodo.7824818)

To manually download the datasets:

1. Click on the provided link above.
2. You will be redirected to the Zenodo page hosting the datasets.
3. On the Zenodo page, click the **Download** button or select specific files to download as needed.
4. Once downloaded, extract the files (if compressed) to a directory of your choice on your local machine.

### Automatic Download with IntelliGraphs

Alternatively, you can use the `IntelliGraphsDataLoader` class to download and prepare the datasets automatically. This method is convenient if you want to streamline the process and ensure that all required data is correctly organized.

To download datasets automatically:

1. Ensure you have the necessary dependencies installed, including `intelligraphs`.

2. Use the following code snippet to download and load the dataset:

    ```python
    from your_project import IntelliGraphsDataLoader  # Replace with the actual import path

    # Initialize the data loader with the desired dataset name
    dataset_name = 'syn-paths'  # Example dataset name, replace with the dataset you want to download
    data_loader = IntelliGraphsDataLoader(dataset_name)

    # Load data into PyTorch DataLoader objects
    train_loader, valid_loader, test_loader = data_loader.load_torch(batch_size=32)
    ```

3. The dataset will be automatically downloaded and extracted to the `.data` directory if it does not already exist.

4. The data will be loaded into PyTorch `DataLoader` objects (`train_loader`, `valid_loader`, `test_loader`) for easy use in training and evaluation.

5. If you prefer to download the dataset only (without loading into PyTorch), simply instantiate the `IntelliGraphsDataLoader` class and it will handle the download and extraction automatically:

    ```python
    from your_project import IntelliGraphsDataLoader  # Replace with the actual import path

    # Initialize the data loader to download the dataset
    dataset_name = 'syn-paths'  # Example dataset name, replace with the dataset you want to download
    data_loader = IntelliGraphsDataLoader(dataset_name)
    ```
The dataset will be saved in the `.data` directory by default.

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

## IntelliGraphs Synthetic KG Generator

TODO

## IntelliGraphs Verifier

TODO

## Baseline Implementations

This project includes several baseline implementations that are used for comparison with advanced models.

### Uniform Baseline Model

The uniform baseline model is designed to serve as a simple reference point. 
It applies a random compression strategy to synthetic and real-world datasets. 
You can run this baseline using the following commands:

```bash
python benchmark/experiments/uniform_compression.py
```

It should complete in about a minute, and GPU acceleration is not necessary for this step.

### Probabilistic KGE models

TO DO

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

## Unit tests

To run the unit tests, install pytest and verify installation:
```bash
pip install pytest
pytest --version
```

Execute the units tests using: 
```bash
pytest
```
