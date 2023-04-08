<p  align="center">
    <img src="images/IntelliGraph-logo.png" width="150px;" style="max-width: 100%;  margin-right:10px;">
<p>
<h1 align="center" dir="auto" style="font-size:60px;">
    IntelliGraphs
</h1>

[TODO Ask Paul register dataset with Zenodo and add DOI badge here]

IntelliGraphs is a collection of graph datasets for benchmarking generative models for knowledge graphs. 
These are graphs that are generated according to first order logic rules.

These datasets are intended to be used for benchmarking generative models for knowledge graphs under transductive settings. 
It can also be used as a testbed for developing new generative models.

This library was designed to be extensible to create synthetic datasets with other custom First Order Rules constraints.

## Datasets

Here is a description of the datasets:

| Dataset | Rules | # Nodes | # Edges | # Relations | # Classes | # Train | # Valid | # Test |
|---------|-------------|---------|---------|-------------|-----------|---------|---------|--------|
|syn-paths|-|-|-|-| - |-|-|-|
|syn-tipr|-|-|-|-|-|-|-|-|
|syn-type|-|-|-|-|-|-|-|-|
|syn-nl|-|-|-|-|-|-|-|-|
|-|-|-|-|-|-|-|-|-|
|-|-|-|-|-|-|-|-|-|


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
pip install intelligraphs
```

## Usage

To use IntelliGraphs, simply:
```python
from intelligraphs import IntelliGraphs

# Create an instance of Intelligraphs with 50 random triples
intelligraph = IntelliGraphs(num_triples=50)

# Get the list of triples
triples = intelligraph.triples

# Get the natural language sentences for the triples
sentences = intelligraph.to_natural_language()

# Print the sentences
for sentence in sentences:
    print(sentence)
```

## License
MIT License

Copyright (c) 2023 Thiviyan Thanapalasingam
