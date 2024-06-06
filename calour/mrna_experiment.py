'''
mRNA experiment (:mod:`calour.mrna_experiment`)
=======================================================

.. currentmodule:: calour.mrna_experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   MRNAExperiment
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy

import numpy as np
import matplotlib as mpl

from .experiment import Experiment
from .io import read
from .util import _get_taxonomy_string, _to_list


logger = getLogger(__name__)


class mRNAExperiment(Experiment):
    '''This class stores transcriptomics (mrna) experiment
    Interactive heatmap gene information is obtained through the mrna_calour module

    This is a child class of :class:`.Experiment`.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The expression table for genes. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    description : str
        name of experiment
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`
    databases: iterable of str, optional
        database interface names to show by default in heatmap() function
        by default use 'dbbact'

    Attributes
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The expression table for genes. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or dense numpy array.
    info : dict
        information about the experiment (data md5, filenames, etc.)
    description : str
        name of the experiment
    databases : dict
        keys are the database names (i.e. 'dbbact' / 'gnps')
        values are the database specific data for the experiment (i.e. annotations for dbbact)

    See Also
    --------
    Experiment
    '''
    def __init__(self, *args, databases=(), **kwargs):
        super().__init__(*args, databases=('mrna',), **kwargs)

    def heatmap(self, *args, **kwargs):
        '''Plot a heatmap for the mrna experiment.

        This method accepts exactly the same parameters as input with
        its parent class method and does exactly the sample plotting.

        The only difference is that by default, its color scale is **in
        log** as its `norm` parameter is set to
        `matplotlib.colors.LogNorm()`. It makes more sense to show the
        gene expression abundances in color of log scale since they cover a wide range of magnitudes.
        You can always set it to other scale as
        explained in :meth:`.Experiment.heatmap`.

        Parameters
        ----------

        Keyword Arguments
        -----------------
        %(experiment.heatmap.parameters)s

        See Also
        --------
        Experiment.heatmap
        '''
        # set this default value inside the function instead of on the
        # function API (like the __init__) because we don't wanna to
        # define mpl.colors.LogNorm() on the API; otherwise, vmin and
        # vmax are set the same once for all mRNAExperiment
        # objects (which we don't want) because python initializes
        # the function arguments when it reads in its definition.

        # by default use the log normalization
        if 'norm' not in kwargs:
            kwargs['norm'] = mpl.colors.LogNorm()
        super().heatmap(*args, **kwargs)

    @staticmethod
    def read(**kwargs):
        '''Load an mRNA transcriptomics experiment. calls calour.io.read() providing the correct class parameter (cls=mRNAExperiment).
        by default, the mRNAExperiment table is expected to be tab separated (can modify by the setting data_file_sep parameter),
        and samples are in columns (can modify by setting sample_in_row parameter).
        By default, the data is not normalized. To normalize the per-sample reads to sum X, set normalize=X.
        For more details, see 

        Parameters
        ----------

        Keyword Arguments
        -----------------
        %(io.read.parameters)s

        Returns
        -------
        ca.MRNAExperiment

        See Also
        --------
        calour.io.read
        '''
        if 'data_file_sep' not in kwargs:
            kwargs['data_file_sep'] = '\t'
        if 'sparse' not in kwargs:
            kwargs['sparse'] = False
        if 'sample_in_row' not in kwargs:
            kwargs['sample_in_row'] = False
        if 'normalize' not in kwargs:
            kwargs['normalize'] = None

        dat = read(**kwargs, cls=mRNAExperiment)
        return dat
