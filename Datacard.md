# Data Card for IntelliGraphs
IntelliGraphs is a collection of datasets for benchmarking Knowledge Graph Generation models. 
It consists of three synthetic datasets (`syn-paths`, `syn-tipr`, `syn-types`) and two real-world datasets 
(`wd-movies`, `wd-articles`). There is also a Python package available that loads these datasets and 
verifies new graphs using semantics that was pre-defined for each dataset.

This data card was created for all five datasets in IntelliGraphs, and the card is based on
[Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI](https://dl.acm.org/doi/abs/10.1145/3531146.3533231). 
Since our dataset is not tabular, not parts of the data card applies. The sections that are not applicable to our datasets
were removed.

#### Dataset Link
https://doi.org/10.5281/zenodo.7824818

#### Data Card Author(s)
- **Thiviyan Thanapalasingam, UvA:** (Contributor)
- **Emile van Krieken, VU:** (Contributor)
- **Peter Bloem, VU:** (Contributor)
- **Paul Groth, UvA:** (Contributor)

## Authorship
### Publishers
#### Publishing Organization(s): University of Amsterdam

#### Industry Type(s): Academic - Tech

#### Contact Detail(s)
- **Publishing POC:** Thiviyan Thanapalasingam
- **Affiliation:** University of Amsterdam
- **Contact:** thiviyan.t@gmail.com
- **Website:** https://doi.org/10.5281/zenodo.7824818

### Dataset Owners

#### Author(s)
- Thiviyan Thanapalasingam, UvA
- Emile van Krieken, VU
- Peter Bloem, VU
- Paul Groth, UvA

## Dataset Overview
#### Data Subject(s)

- **Sensitive Data about people**: The datasets do not contain any sensitive data about people.
- **Non-Sensitive Data about people**: `wd-movies` contains public information about actors and directors, and the movies they have worked on. `wd-articles` contains metadata about publications, but it does contain any identifiable information about researchers. The other datasets do not contain any non-sensitive data about people.
- **Data about natural phenomena**: N/A
- **Data about places and objects**: `syn-types` and `syn-paths` mentions places, but does not contain any data about them.
- **Synthetically generated data**: `syn-types`, `syn-paths`, and `syn-tipr` are synthetically generated. The Intelligraphs Python package shows how these datasets were generated.
- **Data about systems or products and their behaviors**: N/A
- **Unknown**: N/A
- **Others (Please specify)**: N/A

#### Content Description

- **syn-paths** contains synthetic path graphs
- **syn-tipr** contains synthetic subgraph  with [time-index person role graph patterns](http://ontologydesignpatterns.org/wiki/Submissions:Time\_indexed\_person\_role)
- **syn-types** contains synthetic graphs with typed entities
- **wd-movies** contains real-world graphs about movies
- **wd-articles** contains real-world graphs about scientific publications

### Sensitivity of Data

This section was not applicable to our datasets.

### Dataset Version and Maintenance
#### Maintenance Status: 
- **Actively Maintained** - No new versions will be made available, but this dataset will be actively maintained, including but not limited to
updates to the data.

#### Version Details
- **Current Version:** 0.1.2

- **Last Updated:** 06/2023

- **Release Date:** 06/2023

#### Maintenance Plan

We have a dedicated GitHub repository for the dataset, where users can raise any technical issues they encounter.
We will update the datasets if there are any technical issues with the data, or if there are any errors in the data.
Any technical issues can be raised as an issue on the GitHub repository, and we will fix it as soon as possible.

**Versioning:** 

- The dataset follows a versioning system to track changes and updates. The version number is incremented when significant modifications are made to the dataset. This ensures transparency and helps users understand the dataset's evolution over time.
- Criteria for versioning the dataset include major updates to the data sources, significant additions or removals of data, changes to the data schema or format, or any modifications that could impact data analysis or interpretation.

**Updates:** 
- We will update the datasets if there are any technical issues with the data, or if there are any errors in the data.

**Errors:** 
- If any errors are identified in the dataset, we will take immediate action to rectify them. Users can report errors through the GitHub repository or other designated channels, and we will investigate and correct the issues as quickly as possible.
- Criteria for refreshing or updating the dataset due to errors involve the severity of the error, the impact on data integrity or analysis, and the urgency of the fix.

**Feedback:** 
- User feedback is highly valuable in improving the dataset. 
- We encourage users to provide feedback, suggestions, or report any issues they encounter. 
- Feedback can be submitted through the GitHub repository or by contacting the POC (contact details provided above).

#### Expected Change(s)

**Updates to Data:** N/A

**Updates to Dataset:** N/A

**Additional Notes:** N/A

## Example of Data Points

We have provided examples above, so we have not included them here.

## Motivations & Intentions
### Motivations
#### Purpose(s)
- Research: The dataset was created to advance research in the field, and should not be used for production purposes.


#### Domain(s) of Application
`Knowledge Graph`, `Graph Generative Models`, `Neurosymbolic methods`

#### Motivating Factor(s)
- Enabling the development of generative models for Knowledge Graphs
- Studying how well semantics are captured by machine learning models
- Enabling the development of Neurosymbolic methods

### Intended Use
#### Dataset Use(s)
- Safe for research use

#### Suitable Use Case(s)
- **Suitable Use Case:** Test the performance of generative models, and study their ability to capture semantics. 

- **Suitable Use Case:** To develop new Neurosymbolic methods.


#### Unsuitable Use Case(s)
- **Unsuitable Use Case:** Any real-world application, as some parts of these datasets are synthetic and does not represent real-world data.

#### Research and Problem Space(s)
- This dataset is intended to study Subgraph Inference for a given Knowledge Graphs. Please see our paper entitled
*"IntelliGraphs: Datasets for Benchmarking Knowledge Graph Generation"* for more detailed problem description.
 

#### Citation Guidelines
This work is currently under review. We will update this section with the citation guidelines once the paper is accepted. 

**BiBTeX:**
```
[ Submission under review. To be made available after acceptance.]
```

## Access, Rentention, & Wipeout
### Access
#### Access Type
- External - Open Access

#### Documentation Link(s)
- Dataset Website URL: https://zenodo.org/deposit/7824818
- GitHub URL: https://github.com/thiviyanT/IntelliGraphs


#### Access Control List(s)

**Access Control List:** Any person can access the dataset through the Zenodo repository.

### Retention
#### Duration
- We hope to maintain this dataset for the foreseeable future. However, we cannot guarantee that this dataset will be maintained indefinitely.

#### Process Guide
- To ensure long-term access to this dataset, we have uploaded it to Zenodo.


## Provenance
### Collection
#### Method(s) Used

- Derived from a data dump
- Artificially Generated

#### Methodology Detail(s)

**Collection Type**: Derived from a data dump

**Source:** For reproducibility, we use a specific Wikidata dump to extract the data, rather than the live version. For both datasets, we use the Wikidata HDT dump from 3 March 2021, available from the HDT website (https://www.rdfhdt.org/datasets/).

**Platform:** Wikidata (https://www.wikidata.org/) is a free and open knowledge base that can be read and edited by both humans and machines. 

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [ - March 2021]

**Primary modality of collection data:** 

- Graph Data

**Update Frequency for collected data:**

- Static

#### Collection Cadence

**Static:** Data was collected once from single or multiple sources.

#### Limitation(s) and Trade-Off(s)

- We use a static data dump, rather than the live version of Wikidata. This means that the data is not up-to-date, and may not reflect the current state of Wikidata. However, this is necessary for reproducibility, as the live version of Wikidata is constantly changing.
- Our synthetic dataset does not represent real-world data, but captures the semantics of real-world data. This is necessary to ensure that the dataset is suitable for benchmarking machine learning models.


## Human and Other Sensitive Attributes

This section was not applicable to our datasets. 

## Extended Use

This section was not applicable to our datasets.

## Transformations

This section was not applicable to our datasets. 

## Annotations & Labeling

This section was not applicable to our datasets.

## Validation Types

#### Method(s)

- Data Type Validation
- Range and Constraint Validation
- Consistency Validation

#### Breakdown(s)

Each dataset was validated using a set of logical rules, which were reported in our paper *"IntelliGraphs: Datasets for Benchmarking
Knowledge Graph Generation"*. The logical rules are expressed in First Order Logic (FOL), and test various 
properties for each dataset. We refer the reader to the paper for more details.
All graphs were validated for logical consistency. This validation was done programmatically using IntelliGraphs Python library.  

The rest of this section about human validators was not applicable to our datasets.

## Sampling Methods

#### Method(s) Used

- Random Sampling
- Stratified Sampling

#### Characteristic(s)

- To generate synthetic datasets, uniform sampling was used to randomly sample entities and relations to generate
subgraphs.
- To generate wikidata datasets, no sampling was used. But we had random sampling criteria to select the entities that had sufficient overlap in the training, validation and test sets.


#### Sampling Criteria
- To generate synthetic datasets, there were no sampling criteria. 
- To generate wikidata datasets, we queried the wikidata for papers and their associated entities, and selected the entities that had sufficient overlap in the training, validation and test sets.

## Known Applications & Benchmarks

These datasets are intended for Subgraph Inference, which is a novel research problem. We are not aware of any 
existing benchmarks for this task. 


