'''
ratio experiment (:mod:`calour.ratio_experiment`)
=======================================================

.. currentmodule:: calour.ratio_experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   RatioExperiment
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
import pandas as pd

from .experiment import Experiment
from .util import _get_taxonomy_string, _to_list


logger = getLogger(__name__)


class RatioExperiment(Experiment):
    '''This class stores log-ratio data and corresponding analysis methods

    This is a child class of :class:`.Experiment`.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The abundance table for OTUs or ASVs. Samples
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
        The abundance table for OTUs or ASVs. Samples
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def heatmap(self, *args, **kwargs):
        '''Plot a heatmap for the ratio experiment.

        This method accepts exactly the same parameters as input with
        its parent class method and does exactly the sample plotting.

        The only difference is that by default, it uses the diverging 'coolwarm' colormap and bad_color set to white.
        You can always set it to other colormap/bad_color as explained in :meth:`.Experiment.heatmap`.

        See Also
        --------
        Experiment.heatmap

        '''
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'coolwarm'
        if 'bad_color' not in kwargs:
            kwargs['bad_color'] = 'white'
        if 'clim' not in kwargs:
            min_val = np.abs(np.nanmin(self.data))
            max_val = np.abs(np.nanmax(self.data))
            clim_val = np.max([max_val, min_val])
            kwargs['clim'] = (-clim_val, clim_val)
        super().heatmap(*args, **kwargs)

    @classmethod
    def from_exp(self, exp, common_field, group_field, value1, value2=None, threshold=None):
        '''Create a RatioExperiment from two groups of samples in an experiment

        Parameters
        ----------
        exp: calour.Experiment
            The experiment to take the 2 groups of samples from.
        common_field: str
            Name of the sample metadata field to calculate the ratio within.
            For example, to calculate the ratio between 2 timepoints for each individual,
            common_field with be 'subject_id'
        group_field: str
            Name of the sample metadata field on which to divide to 2 groups for ratio calculation.
            For example, to calculate the ratio between 2 timepoints for each individual,
            group_field with be 'time_point'
        value1 : str or iterable of str
            Values for field that will be assigned to the first (nominator) sample group
        value2: str or iterable of str or None, optional
            If not None, values for field that will be assigned to the second (denominator) sample group.
            If None, use all samples not in the first sample group as the second sample group.
        threshold: float or None, optional
            If not None, assign each data value<threshold to threshold. If both nominator and denominator are < threshold,
            the resulting ratio will assigned be np.nan.
            For example, in an AmpliconExperiment, ratios calculated between low read numbers are not reliable. So it is advised to set threshold to approx. 10

        Returns
        -------
        RatioExperiment
        The samples are all the unique values of common_field
        The data values for each feature are the ratio of mean data values in samples of value1 in group_field / data values in samples of value2 in group_field
        '''
        new_field_val = '%s / ' % value1
        exp.sparse = False
        value1 = _to_list(value1)
        if value2 is None:
            new_field_val += 'Other'
            value2 = list(set(exp.sample_metadata[group_field].unique()).difference(set(value1)))
        else:
            new_field_val += '%s' % value2
            value2 = _to_list(value2)

        ratio_mat = np.zeros([len(exp.sample_metadata[common_field].unique()), exp.shape[1]])
        # new_sample_metadata = pd.DataFrame()
        new_sample_metadata = pd.DataFrame(columns=exp.sample_metadata.columns)

        # used to count the number of samples that have values in both groups
        # we then keep only these columns in the ratio_mat
        found_indices = []
        for idx, cexp in enumerate(exp.iterate(common_field, axis=0)):
            # cfield = cexp.sample_metadata[common_field].iloc[0]
            group1 = cexp.filter_samples(group_field, value1)
            group2 = cexp.filter_samples(group_field, value2)
            if group1.shape[0] == 0 or group2.shape[0] == 0:
                continue
            # replace all <= threshold values if needed
            if threshold is not None:
                group1.data[group1.data <= threshold] = threshold
                group2.data[group2.data <= threshold] = threshold

            mean_g1 = np.mean(group1.data, axis=0)
            mean_g2 = np.mean(group2.data, axis=0)
            ratios = np.log2(mean_g1 / mean_g2)
            # Take care of features where both nominator and denominator are below the min_threshold. We should get np.nan
            if threshold is not None:
                ratios[np.logical_and(mean_g1 == threshold, mean_g2 == threshold)] = np.nan
            ratio_mat[idx, :] = ratios
            cmetadata = cexp.sample_metadata.iloc[0]
            for ccol in group1.sample_metadata.columns:
                u1 = group1.sample_metadata[ccol].unique()
                u2 = group2.sample_metadata[ccol].unique()
                if len(u1) == 1 and u1 == u2:
                    cmetadata.at[ccol] = u1[0]
                else:
                    cmetadata.at[ccol] = 'NA (multiple values)'
            cmetadata.at[group_field] = new_field_val
            new_sample_metadata = new_sample_metadata.append(cmetadata)
            found_indices.append(idx)

        # keep only samples that were actually added to the ratio_mat
        ratio_mat = ratio_mat[found_indices, :]
        ratio_exp = RatioExperiment(data=ratio_mat, sample_metadata=new_sample_metadata, feature_metadata=exp.feature_metadata, sparse=False,
                                    databases=exp.databases, description=exp.description, info=exp.info)
        return ratio_exp

    def get_sign_pvals(self, alpha=0.1, min_present=5):
        '''Get FDR corrected p-values for rejecting the null hypothesis that the signs of the ratios originate from a p=0.5 binomial distribution.

        The test is performed only on the non nan feature values.

        Parameters
        ----------
        alpha: float, optional
            The required FDR control level
        min_present: int, optional
            The minimal number of samples where the ratio is not nan in order to include in the test
        '''
        exp = self.copy()
        # get rid of bacteria that don't have enough non-zero ratios
        keep = []
        for idx in range(exp.shape[1]):
            cdat = exp.data[:, idx]
            npos = np.sum(cdat > 0)
            nneg = np.sum(cdat < 0)
            if npos + nneg >= min_present:
                keep.append(idx)
        print('keeping %d features with enough ratios' % len(keep))
        exp = exp.reorder(keep, axis='f')
        pvals = []
        esize = []
        for idx in range(exp.data.shape[1]):
            cdat = exp.data[:, idx]
            npos = np.sum(cdat > 0)
            nneg = np.sum(cdat < 0)
            pvals.append(scipy.stats.binom_test(npos, npos + nneg))
            esize.append((npos - nneg) / (npos + nneg))
        # plt.figure()
        # sp = np.sort(pvals)
        # plt.plot(np.arange(len(sp)),sp)
        # plt.plot([0,len(sp)],[0,1],'k')
        reject = multipletests(pvals, alpha=alpha, method='fdr_bh')[0]
        index = np.arange(len(reject))
        esize = np.array(esize)
        pvals = np.array(pvals)
        exp.feature_metadata['esize'] = esize
        exp.feature_metadata['pval'] = pvals
        index = index[reject]
        okesize = esize[reject]
        new_order = np.argsort(okesize)
        new_order = np.argsort((1 - pvals[reject]) * np.sign(okesize))
        newexp = exp.reorder(index[new_order], axis='f', inplace=False)
        print('found %d significant' % len(newexp.feature_metadata))
        return newexp
