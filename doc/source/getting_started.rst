General Calour concepts
=======================
Introduction
------------
Calour is a python/jupyter notebook module for analysis of tabular (sample X feature) data and associated metadata. Initially desgined for microbiome amplicon experiment analysis (where features are bacteria and the table contains the amount of reads for each features in each sample), Calour has been extended to also facilitate analysis of metabolomics data (where features are unique MS1 or MS2 ids).

Calour synchronizes the 2D data table (feature by sample) with the associated sample metadata (i.e. the sample mapping file) and feature metadata (i.e. per feature data such as taxonomy, MZ/RT etc.) and enables various data manipulation functions (such as sorting, clustering, filtering, etc.)

An important feature of calour is an interactive heatmap for exploration of the data. Since typical experiments may contain hundreds of samples and thousands of features, it is difficult to look at all the features/samples in a single static heatmap. The Calour heatmap allows easy zooming/panning, and displays information selected features/samples. This information can also include data from external databases (such as dbBact.org for amplicon experiments or GNPS for metabolomics experiments), which can display additional information about the selected feature.

Main data structures
--------------------
Calour mostly handles **Experiment** data. An Experiment is made of **Samples**, each containing counts of **Features**. For example, in a typical microbiome amplicon Experiment, each Sample is a swab from an individual, and Features are the unique bacteria present in the Samples.

Calour stores an Experiment as a synchronized set of: per-Sample metadata table (i.e. age, material, name, etc.), per-Feature metadata table (i.e. taxonomy, etc.) and a (sparse or dense) data matrix where position (i,j) stores the number of times (or frequency) feature i was observed in sample j.

Calour contains severl functions for loading (or generating) such an Experiment. Additionally, Calour contains functions for filtering/reordering and statistical analysis of the Samples and Features. Finally, Calour can plot interactive heatmaps for exploring the Experiment and interfacing external databases.

Heatmap GUI
-----------
Calour can display the interactive heatmap either as a stand alone Qt5 window, or as an integrated jupyter notebook figure. The choice of the GUI is made in the `Experiment.plot()` function using the `gui='qt5'` or `gui='jupyter'` options. Additionaly, heatmaps can be generated as a non-interactive figure using the `gui='cli'` option.
