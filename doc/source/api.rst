Using Calour functions via the module and the Experiment class
==============================================================
Calour contains functions that operate on the
``Experiment`` (or its child classes) object. They can be called in
two manners equivalently:

1) called as normal functions with an ``Experiment`` object as first parameter;

2) called as a method upon the ``Experiment`` object.

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

Main classes and utility functions
==================================

.. toctree::
   :maxdepth: 1

   experiment
   amplicon_experiment
   ms1_experiment
   io
   util

Functions operating on ``Experiment`` object
============================================

.. toctree::
   :maxdepth: 1

   heatmap
   plotting
   filtering
   sorting
   transforming
   analysis
   manipulation
   export_html
   training
   database
