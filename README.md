<p  align="center">
    <img src="images/IntelliGraph-logo.png" width="450px;" style="max-width: 100%;  margin-right:10px;">
<p>

---

IntelliGraphs is a library that generates a collection of datasets for benchmarking generative models for knowledge
graphs. These are graphs that are generated according to first order logic rules. These datasets are intended to be used
for benchmarking machine learning models under transductive settings. It can also be used as a testbed for developing
new generative models. This library was designed to be extendable to create new synthetic datasets with other custom 
First-Order Logical (FOL) constraints.

### TODO

* Ask Paul: Do we want to register this with Zenodo. If so, add DOI badge here.
* Ask Paul: How to generate dataset metadata? Is it needed?
* Ask Paul: When to register dataset? Before or after publication?
* Ask Paul: Where to put the dataset so that it lasts? (Zenodo, GitHub, etc.)
* Ask Peter/Paul: Do we want to make it available on PyPI? If so, add badge here.
* Make GitHub repo anonymous before submission

## Advantages

* Easy to use: Generate and manipulate Knowledge Graphs with a simple and clean Python API.
* Flexible: Customize the number of graphs, triples, and data splits.
* Extendable: Create more graphs according to custom FOL rules.
* Efficient: Fast and memory-efficient graph generation and manipulation using native Python data structures.
* Visualization: Visualize Knowledge Graphs.

## Datasets

Here is a description of the datasets:

| Dataset | Rules | # Nodes | # Edges | # Relations | # Classes | # Train | # Valid | # Test |
|---------|-------------|---------|---------|-------------|-----------|---------|---------|--------|
|syn-paths|-|-|-|-| - |-|-|-|
|syn-tipr|-|-|-|-|-|-|-|-|
|syn-type|-|-|-|-|-|-|-|-|
|syn-nl|-|-|-|-|-|-|-|-|
|wd-movies|-|-|-|-|-|-|-|-|
|wd-articles|-|-|-|-|-|-|-|-|

## Example

<table>
  <tr>
    <th>Dataset</th>
    <th>Knowledge Graph</th>
  </tr>
  <tr>
    <td>syn-paths</td>
    <td><pre>
Element_1 has_shape octagon.
    </pre></td>
  </tr>
  <tr>
    <td>syn-tipr</td>
    <td><pre>
Element_1 has_shape octagon.
    </pre></td>
  </tr>
  <tr>
    <td>syn-types</td>
    <td><pre>
Element_1 has_shape octagon.
    </pre></td>
  </tr>
</table>

## Installation

To install IntelliGraphs locally, simply:

```bash
pip install -e .
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

## Future Work

** Inductive Setting** It would be very useful doing the data split such that it allows for inductive setting.

## License

MIT License

Copyright (c) 2023 Thiviyan Thanapalasingam
