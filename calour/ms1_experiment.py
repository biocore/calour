'''
ms1 experiment (:mod:`calour.ms1_experiment`)
=======================================================

.. currentmodule:: calour.ms1_experiment

Classes
^^^^^^^^
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

import numpy as np

from .experiment import Experiment
from .util import _get_taxonomy_string, _to_list


logger = getLogger(__name__)


class MS1Experiment(Experiment):
    '''This class contains the data for a Mass-Spec ms1 spectra experiment or a meta experiment.

    Parameters
    ----------
    data : :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : :class:`pandas.DataFrame`
        The metadata on the samples
    feature_metadata : :class:`pandas.DataFrame`
        The metadata on the features
    description : str
        name of experiment
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`

    Attributes
    ----------
    data : :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : :class:`pandas.DataFrame`
        The metadata on the samples
    feature_metadata : :class:`pandas.DataFrame`
        The metadata on the features
    exp_metadata : dict
        metadata about the experiment (data md5, filenames, etc.)
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or numpy array.
    description : str
        name of the experiment

    See Also
    --------
    Experiment
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmap_feature_field = 'gnps'
        self.heatmap_databases = ('gnps',)

    def __repr__(self):
        '''Return a string representation of this object.'''
        return 'MS1Experiment %s with %d samples, %d features' % (
            self.description, self.data.shape[0], self.data.shape[1])

    def _prepare_gnps(self):
        if '_calour_metabolomics_gnps_table' not in self.exp_metadata:
            logger.warn('No GNPS data file supplied - labels will be NA')
            self.feature_metadata['gnps'] = 'NA'
            return
        logger.debug('Adding gnps terms as "gnps" column in feature metadta')
        self.add_terms_to_features('gnps', use_term_list=None, field_name='gnps')
        logger.debug('Added terms')

    @Experiment._record_sig
    def sort_mz(exp, inplace=False):
        '''Sort the features based on the taxonomy

        Sort features based on the taxonomy (alphabetical)

        Parameters
        ----------
        inplace : bool (optional)
            False (default) to create a copy
            True to Replace data in exp
        Returns
        -------
        exp : Experiment
            sorted by taxonomy
        '''
