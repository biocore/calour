'''
ms1 experiment (:mod:`calour.ms1_experiment`)
=============================================

.. currentmodule:: calour.ms1_experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   MS1Experiment
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger

import matplotlib as mpl
import numpy as np

from .experiment import Experiment
from .util import _to_list

logger = getLogger(__name__)


class MS1Experiment(Experiment):
    '''This class contains the data of Mass-Spec ms1 spectra experiment.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The abundance table for OTUs, metabolites, genes, etc. Samples
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

    Attributes
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or numpy array.
    info : dict
        information about the experiment (data md5, filenames, etc.)
    description : str
        name of the experiment

    See Also
    --------
    Experiment
    '''
    def __init__(self, *args, databases=('gnps',), **kwargs):
        super().__init__(*args, databases=('gnps',), **kwargs)

    def heatmap(self, *args, **kwargs):
        '''Plot a heatmap for the ms1 experiment.

        This method accepts exactly the same parameters as input with
        its parent class method and does exactly the sample plotting.

        The only difference is that by default, its color scale is
        **in log** as its `norm` parameter is set to
        `matplotlib.colors.LogNorm()`. You can always set it to other
        scale as explained in :meth:`.Experiment.heatmap`.

        See Also
        --------
        Experiment.heatmap

        '''
        if 'norm' not in kwargs:
            kwargs['norm'] = mpl.colors.LogNorm()
        super().heatmap(*args, **kwargs)

    def __repr__(self):
        '''Return a string representation of this object.'''
        return 'MS1Experiment %s with %d samples, %d features' % (
            self.description, self.data.shape[0], self.data.shape[1])

    def filter_mz(self, mz, tolerance=0.001, inplace=False, negate=False):
        '''Filter metabolites based on m/z

        Parameters
        ----------
        mz: float or list of float
            the M/Z to filter
        tolerance: float, optional
            the M/Z tolerance. filter metabolites with abs(metabolite_mz - mz) <= tolerance
        inplace: bool, optional
            True to replace current experiment, False to create new experiment with results
        negate: bool, optional
            If False, keep only metabolites matching mz
            If True, remove metabolites matching mz

        Returns
        -------
        calour.MS1Experiment
            features filtered based on mz
        '''
        if 'MZ' not in self.feature_metadata.columns:
            raise ValueError('The Experiment does not contain the column "MZ". cannot filter by mz')
        mz = _to_list(mz)
        keep = set()
        for cmz in mz:
            mzdiff = np.abs(self.feature_metadata['MZ'] - cmz)
            keep = keep.union(set(np.where(mzdiff <= tolerance)[0]))
        if negate:
            keep = set(np.arange(len(self.feature_metadata))).difference(keep)
        return self.reorder(list(keep).sort(), axis='f', inplace=inplace)
