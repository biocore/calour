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

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests

from .experiment import Experiment
from .util import _to_list
from .analysis import _new_experiment_from_pvals, _CALOUR_DIRECTION, _CALOUR_STAT

logger = getLogger(__name__)


class RatioExperiment(Experiment):
    '''This class stores log-ratio data and corresponding analysis methods.

    This is a child class of :class:`.Experiment`.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The log-ratio table for OTUs or ASVs.
        Samples are in rows and features in columns
        Note: values can be negative or np.nan as this is log-ratio.
        np.nan indicates ratio is not applicable between 2 samples for a feature.
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
        The log ratio table for OTUs or ASVs.
        Samples are in row and features in column. values are float (can be negative)
        with np.nan indicating ratio for the specific feature does not exist.
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

        The only difference is that by default, it uses the diverging
        colormap 'coolwarm' and `bad_color` parameter is set to white.  You can
        always set it to other colormap/bad_color as explained in
        :meth:`.Experiment.heatmap`.

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
    def from_exp(self, exp, common_field, group_field, value1, value2=None, threshold=5, sample_md_suffix=None):
        '''Create a RatioExperiment object from two groups of samples in an experiment.

        ratios are calculated for each unique value of common_field. For each such value, 2 groups of samples (group1/group2) are created by selecting all samples with value1/value2
        in the group_field. The ratio for the common field value for each feature is calculated by taking the log2 of mean(group1)/mean(group2) for the feature.

        Parameters
        ----------
        exp: calour.Experiment
            The experiment to take the 2 groups of samples from.
        common_field: str
            Name of the sample metadata field to calculate the ratio within.
            For example, to calculate the ratio between 2 timepoints (e.g. before and after treatment) for each subject,
            common_field would be something like 'subject_id'.
        group_field: str
            Name of the sample metadata field on which to divide samples to 2 groups for ratio calculation.
            For example, to calculate the ratio between before and after treatment for each individual,
            group_field with be 'treatment' that has 'before_treatment' and 'after_treatment'.
        value1 : str or iterable of str
            Values for field that will be assigned to the first (nominator) sample group. For example 'before_treatment'.
            If more than one sample matches value1 in field group_field (for a given common_field value), use the mean of the frequency
            (of each feature) as the nominator value for the sample group.
        value2: str or iterable of str, default=None
            If not None, values for field that will be assigned to the second (denominator) sample group. For example 'after_treatment'
            If None, use all samples not in the first sample group as the second sample group.
            Similar to value1, if more than 1 sample matches, use their mean frequency as the denominator value.
        threshold: float or None, optional
            If not None, assign each data value<threshold to threshold. If both nominator and denominator are < threshold,
            the resulting ratio will assigned be np.nan.
            For example, in an AmpliconExperiment, ratios calculated between low read numbers are not reliable. So it is advised to set threshold to approx. 10
        sample_md_suffix: tuple of (str, str), default=None
            The suffix to add to the ratio experiment for each sample metadata column name, indicating if it is from value1 or value2 group.
            If none, append value1 or value2 to the column name respectively.
            In the ratio experiment, since each sample is the ratio of 2 sample groups, the sample metadata can originate from the first or second group. Therefore
            the 2 values will be stored in 2 columns for each original metadata column.

        Returns
        -------
        RatioExperiment
            The samples are all the unique values of common_field (that have >= 1 sample matching value1 and value2 in group_field).
            The data values for each feature are the ratio of mean data values in samples of value1 in
            group_field / data values in samples of value2 in group_field
            sample_metadata contains two columns for each original sample_metadata column: the value for group1 and for group2
            The index for each ratio sample is the index of the group1 sample
        '''
        new_field_val = '%s / ' % value1
        exp.sparse = False
        value1 = _to_list(value1)
        if value2 is None:
            new_field_val += 'Other'
            value2 = [str(x) for x in set(exp.sample_metadata[group_field].unique()).difference(set(value1))]
        else:
            new_field_val += '%s' % value2
            value2 = _to_list(value2)

        ratio_mat = np.zeros([len(exp.sample_metadata[common_field].unique()), exp.shape[1]])
        # new_sample_metadata = pd.DataFrame()
        if sample_md_suffix is None:
            sample_md_suffix = (".".join(value1), ".".join(value2))
        new_columns = [x + '_%s' % sample_md_suffix[0] for x in exp.sample_metadata.columns]
        new_columns.extend([x + '_%s' % sample_md_suffix[1] for x in exp.sample_metadata.columns])
        new_sample_metadata = pd.DataFrame(columns=new_columns)

        # used to count the number of samples that have values in both groups
        # we then keep only these columns in the ratio_mat
        found_indices = []
        for idx, (_, cexp) in enumerate(exp.iterate(common_field, axis=0)):
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
            cmetadata = pd.DataFrame(columns=new_columns)

            # the new index for this ratio is the sampleID of the nominator sample
            csamp_id = group1.sample_metadata['_sample_id'][0]

            for ccol in group1.sample_metadata.columns:
                u1 = group1.sample_metadata[ccol].unique()
                if len(u1) == 1:
                    cmetadata.at[csamp_id, ccol + '_%s' % sample_md_suffix[0]] = u1[0]
                else:
                    cmetadata.at[csamp_id, ccol + '_%s' % sample_md_suffix[0]] = 'NA (multiple values)'

                u2 = group2.sample_metadata[ccol].unique()
                if len(u2) == 1:
                    cmetadata.at[csamp_id, ccol + '_%s' % sample_md_suffix[1]] = u2[0]
                else:
                    cmetadata.at[csamp_id, ccol + '_%s' % sample_md_suffix[1]] = 'NA (multiple values)'
            # cmetadata.at[group_field] = new_field_val
            new_sample_metadata = new_sample_metadata.append(cmetadata)
            found_indices.append(idx)

        # keep only samples that were actually added to the ratio_mat
        ratio_mat = ratio_mat[found_indices, :]
        logger.info('Calculated ratios for %d unique sample groups' % len(found_indices))
        ratio_exp = RatioExperiment(data=ratio_mat, sample_metadata=new_sample_metadata, feature_metadata=exp.feature_metadata,
                                    sparse=False, databases=exp.databases, description=exp.description, info=exp.info)
        return ratio_exp

    def get_sign_pvals(self, alpha=0.1, min_present=5):
        '''Get FDR corrected p-values for rejecting the null hypothesis that the signs of the ratios originate from a p=0.5 binomial distribution.

        This test is used in order to identify features that increase/decrease significantly. For example, if the RatioExperiments is created for
        pre- and post-treatment samples of individuals (ratio is pre/post), get_sign_pvals can be used to identify features that significantly
        increase/decrease following the treatment.

        NOTE: The test is performed only on the non nan feature values.

        Parameters
        ----------
        alpha: float, optional
            The required FDR control level
        min_present: int, optional
            The minimal number of samples where the ratio is not nan or zero in order to include in the test.
            Used as filtering to achieve better FDR power (less hypothesis to test)

        Returns
        -------
        RatioExperiment
            Only features with higher than random number of positive or negative ratios.
            Features are sorted by the effect size (and by p-value for similar effect size).
            The feature_metadata contains 4 new fields: '__calour_stat', '_calour_pval', '_calour_qval', '_calour_direction'
            , similar to calour.analysis.diff_abundance().
        '''
        exp = self.copy()

        # need to convert to non-sparse in order to use np.isfinite()
        exp.sparse = False

        keep = []
        pvals = np.ones(exp.shape[1])
        esize = np.zeros(exp.shape[1])
        npos = np.zeros(exp.shape[1])
        nneg = np.zeros(exp.shape[1])
        for idx in range(exp.shape[1]):
            cdat = exp.data[:, idx]
            cnpos = np.sum(cdat[np.isfinite(cdat)] > 0)
            cnneg = np.sum(cdat[np.isfinite(cdat)] < 0)
            npos[idx] = cnpos
            nneg[idx] = cnneg
            # test if we have enough non-zero samples
            if npos[idx] + nneg[idx] >= min_present:
                # calculate the binomial p-value and effect size for the feature
                pvals[idx] = scipy.stats.binom_test(cnpos, cnpos + cnneg)
                esize[idx] = (cnpos - cnneg) / (cnpos + cnneg)
                keep.append(idx)
        logger.debug('keeping %d features with enough ratios' % len(keep))
        exp = exp.reorder(keep, axis='f')
        if len(keep) == 0:
            logger.warning('No significant features found')
            return exp

        pvals = pvals[keep]
        esize = esize[keep]

        # multiple testing correction using Benjamini-Hochberg FDR
        # note we cannot use dsFDR as this is not a 2 group test
        reject, qvals, *_ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        newexp = _new_experiment_from_pvals(exp, None, reject, esize, pvals, qvals)
        # set the effect direction field
        newexp.feature_metadata[_CALOUR_DIRECTION] = ['positive' if x > 0 else 'negative' for x in newexp.feature_metadata[_CALOUR_STAT]]

        logger.info('found %d significant' % len(newexp.feature_metadata))
        return newexp
