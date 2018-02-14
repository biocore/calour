General Calour concepts
=======================
Introduction
------------
Calour is a python/jupyter notebook module for analysis of tabular (sample X feature) data and associated metadata. Initially desgined for microbiome amplicon experiment analysis (where features are bacteria and the table contains the amount of reads for each features in each sample), Calour has been extended to also facilitate analysis of metabolomics data (where features are unique MS1 or MS2 ids).



Main data structures
--------------------
Calour mostly handles **Experiment** data. An Experiment is made of **Samples**, each containing counts of **Features**. For example, in a typical microbiome amplicon Experiment, each Sample is a swab from an individual, and Features are the unique bacteria present in the Samples.

Calour stores an Experiment as a synchronized set of: per-Sample metadata table (i.e. age, material, name, etc.), per-Feature metadata table (i.e. taxonomy, etc.) and a (sparse or dense) data matrix where position (i,j) stores the number of times (or frequency) feature i was observed in sample j.

Calour contains severl functions for loading (or generating) such an Experiment. Additionally, Calour contains functions for filtering/reordering and statistical analysis of the Samples and Features. Finally, Calour can plot interactive heatmaps for exploring the Experiment and interfacing external databases (see **here** for example).


Running Calour
==============
Since calour is a python module, there is no need to run it. Instead, Calour is usually imported into the jupyter notebook you are using for analysis.

For a full GUI using Calour, install `EZCalour <https://github.com/amnona/EZCalour>`_.


Starting a Calour analysis notebook
-----------------------------------
If you installed Calour in a conda environment (usually the environment is named calour), first activate the environment:

```source activate calour``` (in max/linux)

or

```activate calour``` (in windows)

In order to run the jupyter notebook server, change directory to the directory where your analysis notebook is located and then type:

```jupyter notebook```

and within the notebook, just import the Calour module using:

```import calour as ca```

For a generic Calour microbiome analysis notebook that contains the standard analysis workflow, you can download the `simple_microbiome_analysis.ipynb <https://raw.githubusercontent.com/biocore/calour/master/notebooks/demo.ipynb>`_.


