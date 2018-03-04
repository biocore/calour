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

import pandas as pd

from .experiment import Experiment
from .database import _get_database_class


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
        self.heatmap_databases = ('gnps',)

    def __repr__(self):
        '''Return a string representation of this object.'''
        return 'MS1Experiment %s with %d samples, %d features' % (
            self.description, self.data.shape[0], self.data.shape[1])

    def _prepare_gnps_ids(self, mzerr=0.1, rterr=30):
        logger.debug('Locating GNPS ids for metabolites based on MS1 MZ/RT')
        if '_calour_metabolomics_gnps_table' not in self.exp_metadata:
            logger.warn('No GNPS data file supplied - labels will be NA')
            self.feature_metadata['gnps'] = None
            return
        try:
            gnps_class = _get_database_class('gnps', exp=self)
            gnps_ids = {}
            for cmet in self.feature_metadata.index.values:
                cids = gnps_class._find_close_annotation(self.feature_metadata['MZ'][cmet], self.feature_metadata['RT'][cmet], mzerr=mzerr, rterr=rterr)
                gnps_ids[cmet] = cids
            self.feature_metadata['_gnps_ids'] = pd.Series(gnps_ids)
        # if the gnps-calour module is not installed
        except ValueError:
            self.feature_metadata['_gnps_ids'] = None
            logger.warning('gnps-calour module not installed. cannot add gnps ids')

    def _prepare_gnps(self):
        if '_calour_metabolomics_gnps_table' not in self.exp_metadata:
            logger.warn('No GNPS data file supplied - labels will be NA')
            self.feature_metadata['gnps'] = 'NA'
            return
        logger.debug('Adding gnps terms as "gnps" column in feature metadta')
        try:
            self.add_terms_to_features('gnps', use_term_list=None, field_name='gnps')
            logger.debug('Added terms')
        # if the gnps-calour module is not installed
        except ValueError:
            logger.warning('GNPS database not found. GNPS terms not added.')
