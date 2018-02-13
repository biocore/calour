General Calour concepts
=======================
Calour mostly handles **Experiment** data. An Experiment is made of **Samples**, each containing counts of **Features**. For example, in a typical microbiome amplicon Experiment, each Sample is a swab from an individual, and Features are the unique bacteria present in the Samples.

Calour stores an Experiment as a synchronized set of: per-Sample metadata table (i.e. age, material, name, etc.), per-Feature metadata table (i.e. taxonomy, etc.) and a (sparse or dense) data matrix where position (i,j) stores the number of times (or frequency) feature i was observed in sample j.

Calour contains severl functions for loading (or generating) such an Experiment. Additionally, Calour contains functions for filtering/reordering and statistical analysis of the Samples and Features. Finally, Calour can plot interactive heatmaps for exploring the Experiment and interfacing external databases (see **here** for example).


Microbiome analysis - step by step
==================================
pitapita


Metabolomics analysis - step by step
====================================
patapata
