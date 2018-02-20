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
You can install Calour locally following the `instructions <https://github.com/biocore/calour/blob/master/INSTALL.md>`_.

You can also try Calour (**without installing**) on an online `mybinder server <https://mybinder.org/v2/gh/amnona/calour/mybinder>`_.


Getting started
---------------
.. toctree::
   :maxdepth: 2

   getting_started


Tutorials
---------
.. toctree::
   :maxdepth: 2

   tutorials_microbiome
   tutorials_metabolome


The Jupyter notebooks in this tutorial section can be downloaded `here <https://github.com/biocore/calour/tree/master/doc/source/notebooks>`_.


API Documentation
-----------------
.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

