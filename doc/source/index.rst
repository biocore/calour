.. calour documentation master file, created by
   sphinx-quickstart on Wed Jan 11 16:08:35 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Calour
=================

Calour is a python module for processing, analysis and interactive exploration of microbiome (and other matrix form data), incorporating external databases.

We recommend using calour inside a `jupyter notebook <http://jupyter.org/>`_ environment.

For a full graphical user interface (point and click - **no python skills needed**), you can use `EZCalour <https://github.com/amnona/EZCalour>`_.

Things you can do with Calour
-----------------------------
* Read and write micrbiome data (`biom <http://biom-format.org/>`_ tables, `qiime2 <https://qiime2.org/>`_ tables), metabolomics data (MS1 or MS2 bucket tables) or any tabular data, along with associated sample/feature metadata files and associated feature phylogenetic tree.

* Normalize, filter, reorder and cluster your data.

* Permutation based differential abundance testing with powerful `dsFDR <http://msystems.asm.org/content/2/6/e00092-17>`_ correction.

* Interactive heatmap plotting withh convenient zoom, pan, multiple feature selection and information about selected feature/sequence

* Integration with databases (`dbBact <http://dbbact.org>`_, `phenoDB <http://msphere.asm.org/content/2/4/e00237-17>`_, `SpongeEMP <http://www.spongeemp.com/main>`_ for microbiome data, `GNPS <https://gnps.ucsd.edu/ProteoSAFe/static/gnps-splash.jsp>`_ for metabolomics) enables viewing and statistical analysis of database information about the experiment features.


Installing Calour
-----------------
Installation instructions are available for **mac/linux** and for **windows**.

You can also try Calour (**without installing**) on an online `mybinder server <https://mybinder.org/v2/gh/amnona/calour/mybinder>`_.


Getting started
---------------
.. toctree::
   :maxdepth: 1

   getting_started


Tutorials
---------
.. toctree::
   :maxdepth: 2

   microbiome_tutorials
   metabolomics_tutorials
   gene_expression_tutorials

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

