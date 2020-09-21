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
        if 'mz_rt' in self.feature_metadata.columns:
            if 'yticklabel_len' not in kwargs:
                kwargs['yticklabel_len'] = None
            if 'feature_field' not in kwargs:
                kwargs['feature_field'] = 'mz_rt'
            if 'yticklabel_kwargs' not in kwargs:
                kwargs['yticklabel_kwargs'] = {'size': 6, 'rotation': 0}
        super().heatmap(*args, **kwargs)

    def __repr__(self):
        '''Return a string representation of this object.'''
        return 'MS1Experiment %s with %d samples, %d features' % (
            self.description, self.data.shape[0], self.data.shape[1])

    def get_spurious_duplicates(self, mz_tolerance=0.001, rt_tolerance=2, corr_thresh=0.8, inplace=False, negate=False):
        '''Get subgroups of metabolites that are suspected ms1 alignment artifacts.

        The function returns a calour.MS1Experiment with groups of metabolites that (within each group) have similar m/z and rt, and are highly
        correlated/anti-correlated. These are usually due to incorrect feature detection/alignment and can be used to optimize the feature selection parameters.
        correlation could be due to incomplete removal of isotopes or same metabolite in multiple RTs
        anti-correlation could be due to RT drift (splitting of one true metabolite)
        Metabolites in the new experiment are ordered by correlation clusters

        Parameters
        ----------
        mz_tolerance: float, optional
            the M/Z tolerance. Metabolites are similar if abs(metabolite_mz - mz) <= mz_tolerance
        rt_tolerance: float, optional
            the retention time tolerance. Metabolites are similar if abs(metabolite_rt - rt) <= rt_tolerance
        corr_threshold: float, optional
            the minimal (abs) correlation/anti-correlation value in order to call features correlated
        inplace: bool, optional
            True to replace current experiment, False to create new experiment with results
        negate: bool, optional
            If False, keep only metabolites that show a correlation with another metabolite
            If True, remove metabolites showing correlation

        Returns
        -------
        MS1Experiment
            features filtered and ordered basen on m/z and rt similarity and correlation
        '''
        features = self.feature_metadata.copy()
        keep_features = []
        data = self.get_data(sparse=False)
        while len(features) > 0:
            # get the first feature
            cfeature = features.iloc[0]
            features.drop(index=cfeature.name, inplace=True)
            # find all mz/rt neighbors of the feature
            mzdist = np.abs(features['MZ'] - cfeature['MZ'])
            rtdist = np.abs(features['RT'] - cfeature['RT'])
            okf = features[np.logical_and(mzdist <= mz_tolerance, rtdist <= rt_tolerance)]
            if len(okf) == 0:
                continue
            # test the correlation of each neighbor
            odat = data[:, self.feature_metadata.index.get_loc(cfeature.name)]
            ckeep = []
            for cf, *_ in okf.iterrows():
                cdat = data[:, self.feature_metadata.index.get_loc(cf)]
                corrcf = np.corrcoef(odat, cdat)[0, 1]
                if np.abs(corrcf) >= corr_thresh:
                    ckeep.append(cf)
            # store the result and remove all the correlated features from the features left to process
            if len(ckeep) > 0:
                keep_features.append(cfeature.name)
                keep_features.extend(ckeep)
                features.drop(index=ckeep, inplace=True)
        return self.filter_ids(keep_features, negate=negate, inplace=inplace)

    def merge_similar_features(self, mz_tolerance=0.001, rt_tolerance=0.5):
        '''Merge metabolites with similar mz/rt to a single metabolite

        Metabolites are initially sorted by frequency and a greedy clustering algorithm (starting from the highest freq.) is used to join together
        metabolites that are close in m/z and r/t, combining them to a signle metabolite with freq=sum(freq) of all metabolites in the cluster.

        Parameters
        ----------
        mz_tolerance: float, optional
            metabolites with abs(metabolite_mz - mz) <= mz_tolerance are joined
        rt_tolerance: float, optional
            metabolites with abs(metabolite_rt - rt) <= rt_tolerance are joined

        Returns
        -------
        MS1Experiment
            With  close metabolites joined to a single metabolite.
            The m/z and rt of the new metabolite are the m/z and rt of the highest freq. metabolite. Frequency of the new metabolite is the sum of frequencies
            of all joined metabolites.
            New feature_metadata fields: _calour_merge_number, _calour_merge_ids are added listing the number and ids of the metabolites joined for each new metabolite
        '''
        exp = self.sort_abundance(reverse=False)
        features = exp.feature_metadata
        features['_metabolite_group'] = np.zeros(len(features)) - 1
        gpos = list(features.columns).index('_metabolite_group')
        cgroup = 0
        for cgroup, cfeature in features.iterrows():
            mzdist = np.abs(features['MZ'] - cfeature['MZ'])
            rtdist = np.abs(features['RT'] - cfeature['RT'])
            ok = (mzdist <= mz_tolerance) & (rtdist <= rt_tolerance) & (features['_metabolite_group'] == -1)
            okpos = np.where(ok)[0]
            for cpos in okpos:
                features.iat[cpos, gpos] = cgroup
        exp = exp.aggregate_by_metadata('_metabolite_group', agg='sum', axis='f')
        exp.feature_metadata.drop('_metabolite_group', axis='columns', inplace=True)
        logger.info('%d metabolites remaining after merge' % len(exp.feature_metadata))
        return exp

    def filter_mz_rt(self, mz=None, rt=None, mz_tolerance=0.05, rt_tolerance=0.2, inplace=False, negate=False):
        '''Filter metabolites based on m/z and/or retention time

        Keep (or remove if negate=True) metabolites that have an m/z and/or retention time close (up to tolerance)
        to the requested mz and/or rt (or list of mz and/or rt).
        If both mz and rt are provided, they should be matched (i.e. filtering is performed using each mz and rt pair with same index)

        Parameters
        ----------
        mz: float or list of float or None, optional
            the M/Z to filter
            if None, do not filter based on M/Z
        rt: float or list of float or None, optional
            the retention time to filter
            if None, do not filter based on rt
        mz_tolerance: float, optional
            the M/Z tolerance. filter metabolites with abs(metabolite_mz - mz) <= mz_tolerance
        rt_tolerance: float, optional
            the rt tolerance. filter metabolites with abs(metabolite_rt - rt) <= rt_tolerance
        inplace: bool, optional
            True to replace current experiment, False to create new experiment with results
        negate: bool, optional
            If False, keep only metabolites matching mz
            If True, remove metabolites matching mz

        Returns
        -------
        MS1Experiment
            features filtered based on mz
        '''
        if mz is None and rt is None:
            raise ValueError('at least one of "mz" and "rt" must not be None')
        if mz is not None:
            if 'MZ' not in self.feature_metadata.columns:
                raise ValueError('The Experiment does not contain the column "MZ". cannot filter by mz')
            else:
                mz = _to_list(mz)
        if rt is not None:
            if 'RT' not in self.feature_metadata.columns:
                raise ValueError('The Experiment does not contain the column "RT". cannot filter by rt')
            else:
                rt = _to_list(rt)

        select = np.zeros(len(self.feature_metadata), dtype='?')
        notfound = 0
        if mz is None:
            mz = [None] * len(rt)
        if rt is None:
            rt = [None] * len(mz)
        if len(mz) != len(rt):
            raise ValueError('mz and rt must have same length')

        for cmz, crt in zip(mz, rt):
            if cmz is not None:
                mzdiff = np.abs(self.feature_metadata['MZ'] - cmz)
                keepmz = mzdiff <= mz_tolerance
            else:
                keepmz = np.full([len(self.feature_metadata)], True)
            if crt is not None:
                rtdiff = np.abs(self.feature_metadata['RT'] - crt)
                keeprt = rtdiff <= rt_tolerance
            else:
                keeprt = np.full([len(self.feature_metadata)], True)
            bothok = np.logical_and(keepmz, keeprt)
            if bothok.sum() == 0:
                notfound += 1
            select = np.logical_or(select, bothok)

        logger.info('Total from mz/rt list with no match: %d' % notfound)
        logger.info('found %d matching features' % np.sum(select))
        if negate:
            select = np.logical_not(select)
        return self.reorder(select, axis='f', inplace=inplace)

    def sort_mz_rt(self, inplace=False):
        '''Sort features according to m/z and retention time.

        This is a convenience function wrapping calour.sort_by_metadata()

        Parameters
        ----------
        inplace: bool, optional
            True to replace current experiment, False to create new experiment with results

        Returns
        -------
        MS1Experiment
            Sorted according to m/z and retention time
        '''
        return self.sort_by_metadata('mz_rt', axis='f', inplace=inplace)
