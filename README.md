<p  align="center">
    <img src="images/IntelliGraph-logo.png" width="150px;" style="max-width: 100%;  margin-right:10px;">
<p>
<h1 align="center" dir="auto" style="font-size:60px;">
    IntelliGraphs
</h1>

IntelliGraphs is a collection of graph datasets for benchmarking generative models for knowledge graphs.

These datasets are intended to be used for benchmarking generative models for knowledge graphs. 
It can also be used as a testbed for developing new generative models for knowledge graphs.

## Datasets

Here is a description of the datasets:

| Dataset | Description | # Nodes | # Edges | # Relations | # Classes | # Train | # Valid | # Test |
|---------|-------------|---------|---------|-------------|-----------|---------|---------|--------|
|syn-paths|-|-|-|-| - |-|-|-|
|syn-tipr|-|-|-|-|-|-|-|-|
|syn-type|-|-|-|-|-|-|-|-|
|syn-nl|-|-|-|-|-|-|-|-|
|-|-|-|-|-|-|-|-|-|
|-|-|-|-|-|-|-|-|-|
    



## Installation

To install IntelliGraphs locally, simply:

```bash
pip install intelligraphs
```

## Usage

To use IntelliGraphs, simply:
```python
from intelligraphs import Intelligraphs

# Create an instance of Intelligraphs with 50 random triples
intelligraph = Intelligraphs(num_triples=50)

# Get the list of triples
triples = intelligraph.triples

# Get the natural language sentences for the triples
sentences = intelligraph.to_natural_language()

# Print the sentences
for sentence in sentences:
    print(sentence)

```

Requirements: 
- Write a simple pip installable library called 'intelligraphs'. Always include docstrings and typing for all functions. This library generates synthetic Knowledge Graphs by randomly triples.
- Generate a Python class that generates synthetic Knowledge Graphs by randomly triples.
- Now modify this class to also return a text that expresses each triples in natural language.
- Now modify the class to include a function that only generates a knowledge graph according to certain first order logic.
- train, valid, test splits