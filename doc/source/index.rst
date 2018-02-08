.. calour documentation master file, created by
   sphinx-quickstart on Wed Jan 11 16:08:35 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Calour
=================

Calour is a python module for processing, analysis and interactive exploration of microbiome (and other matrix form data), incorporating external databases.

We recommend using calour inside a **jupyter notebook** environment.

For a full graphical user interface (point and click - **no python skills needed**), you can use **EZCalour**

The full per-function python API documentation is available **here**

Things you can do with Calour
-----------------------------
* Read and write micrbiome data (biom tables, qiime2 tables), metabolomics data (MS1 or MS2 bucket tables) or any tabular data, along with associated sample/feature metadata files and associated feature phylogenetic tree.

* Normalize, filter, reorder and cluster your data.

* Permutation based differential abundance testing with powerful dsFDR correction.

* Interactive heatmap plotting withh convenient zoom, pan, multiple feature selection and information about selected feature/sequence

* Integration with databases (dbBact.org, phenoDB, SpongeEMP for microbiome, GNPS for metabolomics) enables viewing and statistical analysis of database information about the experiment features.


Installing Calour
-----------------
Installation instructions are available for **mac/linux** and for **windows**

You can also try Calour (without installing) on an online **mybinder server**


General Calour concepts
-----------------------
Calour mostly handles **Experiment** data. An Experiment is made of **Samples**, each containing counts of **Features**. For example, in a typical microbiome amplicon Experiment, each Sample is a swab from an individual, and Features are the unique bacteria present in the Samples.

Calour stores an Experiment as a synchronized set of: per-Sample metadata table (i.e. age, material, name, etc.), per-Feature metadata table (i.e. taxonomy, etc.) and a (sparse or dense) data matrix where position (i,j) stores the number of times (or frequency) feature i was observed in sample j.

Calour contains severl functions for loading (or generating) such an Experiment. Additionally, Calour contains functions for filtering/reordering and statistical analysis of the Samples and Features. Finally, Calour can plot interactive heatmaps for exploring the Experiment and interfacing external databases (see **here** for example).


Usage/Analysis examples
-----------------------

Microbiome
----------
   * Loading and processing a simple microbiome experiment
   * Differential abundance analysis
   * Using dbBact for advanced microbiome analysis
   * Filtering and reordering
   * Normalization
   * Using databases

Mass-Spec
---------

Gene expression
---------------


.. sectnum::

Key classes and utility functions
---------------------------------

.. toctree::
   :maxdepth: 1

   experiment
   amplicon_experiment
   io
   util

Functions operating on ``Experiment`` object
--------------------------------------------

.. toctree::
   :maxdepth: 1

   heatmap
   plotting
   filtering
   sorting
   transforming
   analysis
   manipulation
   training
   database

The above modules contain functions that operate on the
``Experiment`` (or its child classes) object. They can be called in
two manners equivalently: 1) called as normal functions with an
``Experiment`` object as first parameter; 2) called as a method
upon the ``Experiment`` object.

For example:

   >>> from calour import Experiment
   >>> exp = Experiment(np.array([[1,2], [3, 4]]), sparse=False,
   ...                  sample_metadata=pd.DataFrame({'category': ['A', 'B'],
   ...                                                'ph': [6.6, 7.7]},
   ...                                               index=['s1', 's2']),
   ...                  feature_metadata=pd.DataFrame({'motile': ['y', 'n']}, index=['otu1', 'otu2']))

Let's filter samples:

   >>> new1 = exp.filter_samples('category', 'A')
   >>> new1
   Experiment
   ----------
   data dimension: 1 samples, 2 features
   sample IDs: Index(['s1'], dtype='object')
   feature IDs: Index(['otu1', 'otu2'], dtype='object')

Equivalently, we can filter in this way:

   >>> from calour.filtering import filter_samples
   >>> new2 = filter_samples(exp, 'category', 'A')
   >>> new2
   Experiment
   ----------
   data dimension: 1 samples, 2 features
   sample IDs: Index(['s1'], dtype='object')
   feature IDs: Index(['otu1', 'otu2'], dtype='object')
   >>> new1 == new2
   True


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

