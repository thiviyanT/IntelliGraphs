<p  align="center">
    <img src="assets/IntelliGraphs-logo 222x222(32-bit).png" width="150px;" style="max-width: 100%;  margin-right:10px;">
    <h1 align="center" >
        IntelliGraphs: Benchmark Datasets for Knowledge Graph Generation
    </h1>
<p>

<p align=center>
    <a href="https://pypi.org/project/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=PyPI%20package&logo=pypi&logoColor=white" title="The current version of IntelliGraphs"></a>
    <a href="https://anaconda.org/thiv/intelligraphs/"><img src="https://img.shields.io/pypi/v/intelligraphs?color=%23099cec&label=Anaconda.org&logo=anaconda&logoColor=white" title="IntelliGraphs on Conda"></a>
    <a href="https://github.com/intelligraphs/layout-parser/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC--BY-blue.svg" title="IntelliGraphs uses CC-BY License"></a>
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/intelligraphs">
</p>

<p align=center>
    <a href="https://doi.org/10.5281/zenodo.14787483"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.14787483.svg" title="DOI of IntelliGraphs"></a>
    <a href="https://arxiv.org/abs/2307.06698"><img src="https://img.shields.io/badge/preprint-2307.06698-b31b1b.svg" title="IntelliGraphs Paper"></a>
    <a href="https://github.com/thiviyanT/IntelliGraphs/actions/workflows/test.yml"><img src="https://github.com/thiviyanT/IntelliGraphs/actions/workflows/test.yml/badge.svg" title="Tests"></a>
    <a href="https://github.com/thiviyanT/IntelliGraphs/actions"><img src="https://github.com/thiviyanT/IntelliGraphs/actions/workflows/publish.yml/badge.svg" title="IntelliGraphs Build Status"></a>
<!---
    <a href="https://intelligraphs.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/doc-layout--parser.readthedocs.io-light.svg" title="IntelliGraphs Documentation"></a>
--->
</p>

IntelliGraphs is a collection of benchmark datasets specifically for use in 
benchmarking generative models for knowledge graphs. You can learn more about it in our preprint: 
[*IntelliGraphs: Datasets for Benchmarking Knowledge Graph Generation*](https://arxiv.org/abs/2307.06698). The Python package provides easy access to the datasets, along with pre- and post-processing functions, 
baseline models, and evaluation tools for benchmarking new models.


## Installation

IntelliGraphs can be installed using either `pip` or `conda`. Dependencies be automatically installed during the 
installation process. 

##### Install with pip:
```bash 
pip install intelligraphs         # Standard pip
uv pip install intelligraphs     # Using UV (faster)
```
##### Install with conda:
```bash 
conda install -c thiv intelligraphs
```

#### Verifying the Installation

After installation, you can verify that IntelliGraphs has been successfully installed by running the following command in your Python environment:

```bash
python -c "import intelligraphs; print(intelligraphs.__version__)"
```

It is recommended to use the latest version. If you don't have the latest version (check badge above), please ensure to update your installation before using it: 

```bash
pip install --upgrade intelligraphs  # or conda install -c thiv intelligraphs --upgrade
```

## Downloading the Datasets

The datasets required for this project can be obtained either manually or automatically through IntelliGraphs Python package.

### Manual Download

The datasets are hosted on Zenodo: [https://doi.org/10.5281/zenodo.14787483](https://doi.org/10.5281/zenodo.14787483)

You can download the datasets and extract the files to your preferred directory.

### Automatic Dataset Download  

To download, verify, and extract datasets automatically, use:

```bash
python -m intelligraphs.data_loaders.download
```
This command will download all IntelliGraphs datasets, verify their integrity using MD5 checksums, and then extract 
them into the `.data` directory in your current working directory.

## IntelliGraphs Data Loader

The `DataLoader` class is a utility for loading IntelliGraphs datasets, simplifying the process of accessing and organizing the data for machine learning tasks. It provides functionalities to download, extract, and load the datasets into PyTorch tensors.

### Usage
1. Instantiate the DataLoader:
``` python
from intelligraphs import DataLoader
data_loader = DataLoader(dataset_name='syn-paths')
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

### `SynPathsGenerator`

This generator creates path graphs where each node represents a city in the Netherlands and each edge represents a mode of transport (`cycle_to`, `drive_to`, `train_to`).

- **Entities:** Dutch cities
- **Relations:** Modes of transport between cities
- **Use case:** Structural learning

### `SynTIPRGenerator`

This generator creates graphs representing academic roles, timelines, and people. The nodes represent individuals, roles, and years, and the edges represent relationships like `has_name`, `has_role`, `start_year`, and `end_year`.

- **Entities:** Names, roles, years
- **Relations:** Relationships between academic roles and timeframes
- **Use case:** Basic temporal reasoning and type checking

### `SynTypesGenerator`

This generator creates graphs where nodes represent countries, languages, and cities, and edges represent relationships like `spoken_in`, `part_of`, and `same_as`.

- **Entities:** Countries, languages, cities
- **Relations:** Geographical and linguistic relationships
- **Use case:** Type checking

### Customization

Each generator class inherits from `BaseSyntheticDatasetGenerator` and can be customized by overriding methods or adjusting parameters. The base class provides utility methods for splitting datasets, checking for unique graphs, and visualizing graphs.

#### Extending Functionality

To create a new dataset generator, simply create a new class that inherits from `BaseSyntheticDatasetGenerator` and implement the `sample_synthetic_data` method to define your dataset's logic.

```python
class MyCustomDatasetGenerator(BaseSyntheticDatasetGenerator):
    def sample_synthetic_data(self, num_graphs):
        # Implement your custom logic here
        pass
```

### Data Generation

You can generate synthetic datasets by running the corresponding script for 
each generator. Each generator allows customization of dataset size, random 
seed, and other parameters.

```bash
python intelligraphs/generator/synthetic/synpaths_generator.py --train_size 60000 --val_size 20000 --test_size 20000 --num_edges 3 --random_seed 42 --dataset_name "syn-paths"
python intelligraphs/generator/synthetic/syntypes_generator.py  --train_size 60000 --val_size 20000 --test_size 20000 --num_edges 3 --random_seed 42 --dataset_name "syn-types"
python intelligraphs/generator/synthetic/syntipr_generator.py --train_size 50000 --val_size 10000 --test_size 10000 --num_edges 3 --random_seed 42 --dataset_name "syn-tipr"
```

## IntelliGraphs Verifier

### Rules

Every dataset comes with a set of rules that describe the nature of the graphs.  The `ConstraintVerifier` class includes a convenient method called `print_rules()` that allows you to view all the rules and their descriptions in a clean and organized format.

To use the `print_rules()` method, simply instantiate a subclass of `ConstraintVerifier`, such as `SynPathsVerifier`, and then call the `print_rules()` method on that instance to list the logical rules for a given dataset.

#### Example Usage

```python
from intelligraphs.verifier.synthetic import SynPathsVerifier

# Initialize the verifier for the syn-paths dataset
verifier = SynPathsVerifier()

# Print the rules and their descriptions for the syn-paths dataset
verifier.print_rules()
```

When you call `print_rules()`, you'll get a formatted list of all the rules along with their corresponding descriptions. For example:

```
List of Rules and Descriptions:
        -> Rule 1:
           FOL: ∀x, y, z: connected(x, y) ∧ connected(y, z) ⇒ connected(x, z)
           Description: Ensures transitivity. If x is connected to y, and y is connected to z, then x should be connected to z.
        -> Rule 2:
           FOL: ∀x, y: edge(x, y) ⇒ connected(x, y)
           Description: If there's an edge between two nodes x and y, then x should be connected to y.
        ...
```

## Baseline Models

### Importing Baseline Models

Our baseline models are also available through the Python API. 
You can find them inside [baseline_models](./intelligraphs/baseline_models) class.

To import the Uniform Baseline model:

```python
from intelligraphs.baseline_models import UniformBaseline
```

To import the Knowledge Graph Embedding (KGE) models:

```python
from intelligraphs.baseline_models.knowledge_graph_embedding_model import KGEModel
```


### Setup

To recreate our experiments, we recommend using a fresh virtual environment with Python 3.10 installed. 

#### 1. Install package
```bash
pip install -e .  # or: pip install intelligraphs  # or: conda install -c thiv intelligraphs
```

#### 2. Install dependencies
```bash
pip install torch pyyaml tqdm wandb numpy scipy
```

#### 3. Configure tracking
```bash
wandb login  # or disable with: export WANDB_MODE=disabled
```
### Uniform Baseline Model

The uniform baseline model is designed to serve as a simple reference baseline. 
It applies a random compression strategy for the synthetic and real-world datasets. 
You can run this baseline using the following commands:

```bash
python benchmark/experiments/uniform_baseline_compression_test.py
```

It should complete in about a minute without any GPU-acceleration. 

To run the graph sampling experiment using the uniform sampler, run the command:
```bash
python benchmark/experiments/uniform_baseline_graph_sampling.py
```

### Probabilistic KGE Models

We've developed three CUDA-compatible probabilistic Knowledge Graph Embedding models: [TransE](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf), [DistMult](https://arxiv.org/abs/1412.6575), and [ComplEx](https://proceedings.mlr.press/v48/trouillon16.pdf). Run experiments using the commands below:

#### Synthetic Datasets
```bash
# syn-paths
python benchmark/experiments/probabilistic_kge_baselines.py --config benchmark/configs/syn-paths-[model].yaml

# syn-tipr
python experiments/train_baseline.py --config benchmark/configs/syn-tipr-[model].yaml

# syn-types
python benchmark/experiments/probabilistic_kge_baselines.py --config benchmark/configs/syn-types-[model].yaml
```

#### Wikidata Datasets
```bash
# wd-articles and wd-movies
python benchmark/experiments/probabilistic_kge_baselines.py --config benchmark/configs/wd-[dataset]-[model].yaml
```

Replace `[model]` with `transe`, `complex`, or `distmult` and `[dataset]` with the appropriate dataset name.

## Dataset Verification

We have written test functions to check the graphs in the datasets against the list of rules. It can be run using: 
```bash
python intelligraphs/data_validation/validate_data.py
```

If there are any errors in the data, it will raise a `DataError` exception and the error message will look similar to this:
```
intelligraphs.errors.custom_error.DataError: Violations found in a graph from the training dataset: 
        - Rule 6: An academic's tenure end year cannot be before its start year. The following violation(s) were found: (_time, start_year, 1996), (_time, end_year, 1994).
```

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

## Reporting Issues

If you encounter any bugs or have any feature requests, please file an issue [here](https://github.com/thiviyanT/IntelliGraphs/issues).

## License

IntelliGraphs datasets and the python package is licensed under CC-BY License. See [LICENSE](LICENSE) for more information.

## Platform Compatibility/Issues

This package has been and developed and tested on MacOS and Linux operating systems. 
If you experience any problems on Windows or any other issues, please [raise the issue on the project's GitHub repository](https://github.com/thiviyanT/IntelliGraphs/issues).

## Unit tests

Make sure to activate the virtual environment with the installation of the intelligraphs package.

To run the unit tests, install pytest:

`pip install pytest` or `conda install pytest`

```bash
pytest --version  # verify installation
```

Execute the units tests using: 
```bash
pytest
```

## Contributing

If you would like to contribute code for a new feature or bug fix, here's how to get started:

First, set up your development environment:
```bash
git clone https://github.com/thiviyanT/IntelliGraphs.git
cd IntelliGraphs

python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install development dependencies
pip install -e .
```

For submitting changes:
```bash
# Create a new branch from dev
git checkout dev
git checkout -b feature/your-feature-name

# Make your changes and commit
git add .
git commit -m "Description of your changes"

# Push to GitHub
git push -u origin feature/your-feature-name
```

To submit changes:
1. Ensure all tests pass by running pytest.
2. Update the README.md, if needed.
3. Create a pull request from your feature branch to the dev branch.
4. The CI pipeline will automatically run tests on your pull request.

Changes must pass all tests and be approved before they can be merged into the `main` branch. 
For questions or discussions, please open an issue on GitHub.
