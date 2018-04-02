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

    def _prepare_gnps_ids(self, direct_ids=False, mz_thresh=0.02, rt_thresh=15, use_gnps_id_from_AllFiles=True):
        '''Link each feature to the corresponding gnps table id.

        Parameters
        ----------
        direct_ids: bool, optional
            True to link via the ids, False (default) to link via MZ/RT
        mz_thresh, rt_thresh: float, optional
            the threshold for linking to gnps if direct_ids is False
        use_gnps_id_from_AllFiles: bool, optional
            True (default) to link using the AllFiles column in GNPS, False to link using 'cluster index' column
            (if direct_ids is True).
        '''
        logger.debug('Locating GNPS ids')
        self.feature_metadata['gnps'] = None
        # if we don't have the linked gnps table, all are NA
        if '_calour_metabolomics_gnps_table' not in self.exp_metadata:
            logger.info('No GNPS data file supplied - gnps labels will be NA')
            return
        gnps_data = self.exp_metadata['_calour_metabolomics_gnps_table']
        if direct_ids:
            # get the gnps ids values from the gnps file
            if use_gnps_id_from_AllFiles:
                gnps_metabolite_ids = []
                for cid in gnps_data['AllFiles']:
                    cid = cid.split(':')[-1]
                    cid = cid.split('###')[0]
                    cid = int(cid)
                    gnps_metabolite_ids.append(cid)
            else:
                gnps_metabolite_ids = gnps_data['cluster index']
            gnps_metabolite_ids_pos = {}
            for idx, cmet in enumerate(gnps_metabolite_ids):
                gnps_metabolite_ids_pos[cmet] = idx
            gnps_ids = {}
            for cmet in self.feature_metadata.index.values:
                if cmet in gnps_metabolite_ids_pos:
                    gnps_ids[cmet] = gnps_metabolite_ids_pos[cmet]
                else:
                    raise ValueError('metabolite ID %s not found in gnps file. Are you using correct ids?' % cmet)
        else:
            # match using MZ/RT
            logger.debug('linking using MZ (thresh %f) and RT (thresh %f)' % (mz_thresh, rt_thresh))
            try:
                gnps_class = _get_database_class('gnps', exp=self)
                gnps_ids = {}
                for cmet in self.feature_metadata.index.values:
                    cids = gnps_class._find_close_annotation(self.feature_metadata['MZ'][cmet], self.feature_metadata['RT'][cmet], mzerr=mz_thresh, rterr=rt_thresh)
                    gnps_ids[cmet] = cids
            # if the gnps-calour module is not installed
            except ValueError:
                logger.warning('gnps-calour module not installed. cannot add gnps ids')
        self.feature_metadata['gnps'] = pd.Series(gnps_ids)

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
