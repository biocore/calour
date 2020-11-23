'''
analysis (:mod:`calour.analysis`)
=================================

.. currentmodule:: calour.analysis

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   correlation
   diff_abundance
   diff_abundance_kw
   diff_abundance_paired
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy as sp

from .experiment import Experiment
from .util import _to_list, format_docstring
from . import dsfdr


logger = getLogger(__name__)


_CALOUR_STAT = '_calour_stat'
_CALOUR_PVAL = '_calour_pval'
_CALOUR_QVAL = '_calour_qval'
_CALOUR_DIRECTION = '_calour_direction'


@format_docstring(_CALOUR_PVAL, _CALOUR_QVAL, _CALOUR_STAT, _CALOUR_DIRECTION)
def correlation(exp: Experiment, field, method='spearman', nonzero=False, transform=None,
                numperm=1000, alpha=0.1, fdr_method='dsfdr', random_seed=None):
    '''Find features with correlation to a numeric metadata field.

    The permutation based p-values and multiple hypothesis correction is implemented.

    Parameters
    ----------
    field: str
        The field to test by. Values are converted to numeric.
    method : str or function
        the method to use for the statistic. options:

        * 'spearman': spearman correlation
        * 'pearson': pearson correlation
        * callable: the callable to calculate the statistic (its input are
          sample-by-feature numeric numpy.array and 1D numeric
          numpy.array of sample metadata; output is a numpy.array of float)
    nonzero : bool, optional
        True to calculate the correlation only for samples where the feature is present (>0).
        False (default) to calculate the correlation over all samples
        Note: setting nonzero to True slows down the calculation
        Note: can be set to True only using 'spearman' or 'pearson', not using a custom function
    transform : str or None
        transformation to apply to the data before caluculating the statistic.

        * 'rankdata' : rank transfrom each OTU reads
        * 'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        * 'normdata' : normalize the data to constant sum per samples
        * 'binarydata' : convert to binary absence/presence
    alpha : float
        the desired FDR control level (type I error rate)
    numperm : int
        number of permutations to perform
    fdr_method : str
        method to compute FDR. Allowed methods include:

        * 'dsfdr': discrete FDR
        * 'bhfdr': Benjamini-Hochberg FDR method
        * 'byfdr' : Benjamini-Yekutielli FDR method
        * 'filterBH' : Benjamini-Hochberg FDR method with filtering
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed for the random permutations
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    Returns
    -------
    Experiment
        The experiment with only correlated features, sorted according to correlation coefficient.

        * '{}' : the non-adjusted p-values for each feature
        * '{}' : the FDR-adjusted q-values for each feature
        * '{}' : the statistics (correlation coefficient if
          the `method` is 'spearman' or 'pearson'). If it
          is larger than zero for a given feature, it indicates this
          feature is positively correlated with the sample metadata;
          otherwise, negatively correlated.
        * '{}' : in which of the 2 sample groups this given feature is increased.
    '''
    if field not in exp.sample_metadata.columns:
        raise ValueError('Field %s not in sample_metadata. Possible fields are: %s' % (field, exp.sample_metadata.columns))

    cexp = exp.filter_sum_abundance(0, strict=True)

    data = cexp.get_data(copy=True, sparse=False).transpose()

    labels = pd.to_numeric(exp.sample_metadata[field], errors='coerce').values
    # remove the nans
    nanpos = np.where(np.isnan(labels))[0]
    if len(nanpos) > 0:
        logger.warning('NaN values encountered in labels for correlation. Ignoring these samples')
        labels = np.delete(labels, nanpos)
        data = np.delete(data, nanpos, axis=1)
    # change the method if we have nonzero
    if nonzero:
        if method == 'spearman':
            method = 'nonzerospearman'
        elif method == 'pearson':
            method = 'nonzeropearson'
        else:
            raise ValueError('Cannot use nonzero=True on methods except "pearson" or "spearman"')
    # find the significant features
    keep, odif, pvals, qvals = dsfdr.dsfdr(data, labels, method=method, transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method, random_seed=random_seed)
    logger.info('Positive correlated features : %d. Negative correlated features : %d. total %d'
                % (np.sum(odif[keep] > 0), np.sum(odif[keep] < 0), np.sum(keep)))
    newexp = _new_experiment_from_pvals(cexp, exp, keep, odif, pvals, qvals)
    newexp.feature_metadata[_CALOUR_DIRECTION] = [field if x > 0 else 'Anti-%s' % field for x in newexp.feature_metadata[_CALOUR_STAT]]
    return newexp


@format_docstring(_CALOUR_PVAL, _CALOUR_QVAL, _CALOUR_STAT, _CALOUR_DIRECTION)
def diff_abundance(exp: Experiment, field, val1, val2=None, method='meandiff', transform='rankdata', numperm=1000, alpha=0.1, fdr_method='dsfdr', shuffler=None, random_seed=None):
    '''Differential abundance test between 2 groups of samples for all the features.

    It uses permutation based nonparametric test and then applies
    multiple hypothesis correction. The idea is that you compute a
    defined statistic and compare it to the distribution of the same
    statistic values computed from many permutations.

    Parameters
    ----------
    field : str
        The field from sample metadata to group samples.
    val1 : str or list of str
        The values in the `field` column for the first group.
    val2 : str or list of str or None (optional)
        The values in the `field` column to select the second group.
        `None` (default) to compare to all other samples (excluding `val1`).
    method : str or function
        the method to compute the statistic. options:

        * 'meandiff' : mean(A)-mean(B)
        * 'stdmeandiff' : (mean(A)-mean(B))/(std(A)+std(B))
        * callable : use this to calculate the statistic (input is data,labels, output is array of float)

    transform : str or None
        transformation to apply to the data before caluculating the statistic.

        * 'rankdata' : rank transfrom each OTU reads
        * 'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        * 'normdata' : normalize the data to constant sum per samples
        * 'binarydata' : convert to binary absence/presence

    alpha : float (optional)
        the desired FDR control level
    numperm : int (optional)
        number of permutations to perform
    fdr_method : str (optional)
        The method used to control the False Discovery Rate. options are:

        * 'dsfdr' : the discrete FDR control method
        * 'bhfdr' : Benjamini-Hochberg FDR method
        * 'byfdr' : Benjamini-Yekutielli FDR method
        * 'filterBH' : Benjamini-Hochberg FDR method following removal
          of all features with minimal possible p-value less than
          alpha (e.g. a feature that appears in only 1 sample can
          obtain a minimal p-value of 0.5 and will therefore be
          removed when say alpha=0.1)
    shuffler: function or None, optional
        if None, use shuffling on all samples (using the random_seed supplied)
        if function, use thi supplied function to shuffle to labels for random iteration. Can be used for paired shuffling, etc.
        Input to the function is the labels (np.array), and the random number generator (np.radnom.Generator), output is the shuffled labels (np.array)
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed for the random permutations
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    Returns
    -------
    Experiment
        A new experiment with only significant features, sorted
        according to their effect size.  The new experiment contains
        additional ``feature_metadata`` fields that include:

        * '{}' : the non-adjusted p-values for each feature
        * '{}' : the FDR-adjusted q-values for each feature
        * '{}' : the effect size (t-statistic). If it is larger than
          zero for a given feature, it indicates this feature is
          increased in the first group of samples (``val1``); if
          smaller than zero, this feature is decreased in the first
          group.
        * '{}' : in which of the 2 sample groups this given feature is increased.
    '''
    if field not in exp.sample_metadata.columns:
        raise ValueError('Field %s not in sample_metadata. Possible fields are: %s' % (field, exp.sample_metadata.columns))

    # if val2 is not none, need to get rid of all other samples (not val1/val2)
    val1 = _to_list(val1)
    grp1 = ','.join(val1)
    if val2 is not None:
        val2 = _to_list(val2)
        cexp = exp.filter_samples(field, val1 + val2, negate=False)
        grp2 = ','.join(val2)
        logger.info('%d samples with both values' % cexp.shape[0])
    else:
        cexp = exp
        grp2 = 'NOT %s' % grp1

    # remove features not present in both groups
    cexp = cexp.filter_sum_abundance(0, strict=True)

    data = cexp.get_data(copy=True, sparse=False).transpose()
    # prepare the labels.
    labels = np.zeros(len(cexp.sample_metadata))
    labels[cexp.sample_metadata[field].isin(val1).values] = 1
    logger.info('%d samples with value 1 (%s)' % (np.sum(labels), val1))
    keep, odif, pvals, qvals = dsfdr.dsfdr(data, labels, method=method, transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method, shuffler=shuffler, random_seed=random_seed)
    logger.info('number of higher in {}: {}. number of higher in {} : {}. total {}'.format(
        grp1, np.sum(odif[keep] > 0), grp2, np.sum(odif[keep] < 0), np.sum(keep)))
    newexp = _new_experiment_from_pvals(cexp, exp, keep, odif, pvals, qvals)
    newexp.feature_metadata[_CALOUR_DIRECTION] = [grp1 if x > 0 else grp2 for x in newexp.feature_metadata[_CALOUR_STAT]]
    return newexp


def diff_abundance_kw(exp: Experiment, field, transform='rankdata', numperm=1000, alpha=0.1, fdr_method='dsfdr', random_seed=None):
    '''Test the differential abundance between multiple sample groups using the Kruskal Wallis test.

    It uses permutation based nonparametric test and then applies
    multiple hypothesis correction.

    Parameters
    ----------
    exp: Experiment
    field: str
        The field to test by
    transform : str or None
        transformation to apply to the data before caluculating the statistic

        * 'rankdata' : rank transfrom each OTU reads
        * 'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        * 'normdata' : normalize the data to constant sum per samples
        * 'binarydata' : convert to binary absence/presence

    alpha : float
        the desired FDR control level
    numperm : int
        number of permutations to perform
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed for the random permutations
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    Returns
    -------
    newexp : Experiment
        The experiment with only significant features, sorted according to difference.

    See Also
    --------
    diff_abundance

    '''
    if field not in exp.sample_metadata.columns:
        raise ValueError('Field %s not in sample_metadata. Possible fields are: %s' % (field, exp.sample_metadata.columns))

    logger.debug('diff_abundance_kw for field %s' % field)

    # remove features with 0 abundance
    cexp = exp.filter_sum_abundance(0, strict=True)

    data = cexp.get_data(copy=True, sparse=False).transpose()
    # prepare the labels. If correlation method, get the values, otherwise the group
    labels = np.zeros(len(exp.sample_metadata))
    for idx, clabel in enumerate(exp.sample_metadata[field].unique()):
        labels[exp.sample_metadata[field].values == clabel] = idx
    logger.debug('Found %d unique sample labels' % (idx + 1))
    keep, odif, pvals, qvals = dsfdr.dsfdr(data, labels, method='kruwallis', transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method, random_seed=random_seed)

    logger.info('Found %d significant features' % (np.sum(keep)))
    return _new_experiment_from_pvals(cexp, exp, keep, odif, pvals, qvals)


@format_docstring(_CALOUR_PVAL, _CALOUR_QVAL, _CALOUR_STAT, _CALOUR_DIRECTION)
def diff_abundance_paired(exp: Experiment, pair_field, field, val1, val2=None, transform='rankdata', random_seed=None, **kwargs):
    '''Differential abundance test between 2 groups of samples for all the features.

    It uses permutation based nonparametric test and then applies
    multiple hypothesis correction. The idea is that you compute a
    defined statistic and compare it to the distribution of the same
    statistic values computed from many permutations.

    Parameters
    ----------
    pair_field: str
        The sample metadata field on which samples are paired.
        NOTE:
        field values with !=2 samples are dropped.
    field : str
        The field from sample metadata to group samples.
    val1 : str or list of str
        The values in the `field` column for the first group.
    val2 : str or list of str or None (optional)
        The values in the `field` column to select the second group.
        `None` (default) to compare to all other samples (excluding `val1`).
    transform: str or None, optional
        Similar to diff_abundance transform parameter. Additional options are:
            'pair_rank': for each group of samples (a single value in pair_field), samples are ranked within the group.

    Keyword Arguments
    -----------------
    %(analysis.diff_abundance.parameters)s

    Returns
    -------
    Experiment
        A new experiment with only significant features, sorted
        according to their effect size.  The new experiment contains
        additional ``feature_metadata`` fields that include:

        * '{}' : the non-adjusted p-values for each feature
        * '{}' : the FDR-adjusted q-values for each feature
        * '{}' : the effect size (t-statistic). If it is larger than
          zero for a given feature, it indicates this feature is
          increased in the first group of samples (``val1``); if
          smaller than zero, this feature is decreased in the first
          group.
        * '{}' : in which of the 2 sample groups this given feature is increased.
    '''
    val1 = _to_list(val1)
    if val2 is not None:
        val2 = _to_list(val2)
        exp = exp.filter_samples(field, val1 + val2, negate=False)

    # keep only paired samples
    drop_values = []
    for cval, cexp in exp.iterate(pair_field):
        if len(cexp.sample_metadata) < 2:
            logger.debug('Value %s has only %d samples. dropped' % (cval, len(cexp.sample_metadata)))
            drop_values.append(cval)
    if len(drop_values) > 0:
        logger.info('Dropping %d values with < 2 samples' % len(drop_values))
        exp = exp.filter_samples(pair_field, drop_values, negate=True)

    # create the groups list for the shuffle function
    groups = defaultdict(list)
    for pos, (idx, crow) in enumerate(exp.sample_metadata.iterrows()):
        groups[crow[pair_field]].append(pos)
    if transform == 'pair_rank':
        # copy so we don't change the original experiment
        exp = exp.copy()
        # make all pairs to 0/1 (low/high) for each feature
        exp.sparse = False
        for cval in exp.sample_metadata[pair_field].unique():
            cpos = np.where(exp.sample_metadata[pair_field] == cval)[0]
            cdat = exp.data[cpos, :]
            exp.data[cpos, :] = sp.stats.rankdata(cdat, axis=0)
        # no need to do another transform in diff_abundance
        transform = None

    # create the numpy.random.Generator for the paired shuffler
    rng = np.random.default_rng(random_seed)

    def _pair_shuffler(labels, rng=rng, groups=groups):
        clabels = labels.copy()
        for cgroup in groups.values():
            clabels[cgroup] = rng.permutation(clabels[cgroup])
        return clabels

    # # sort by pairing (so the pair shuffler will work)
    # exp = exp.sort_samples(pair_field)
    newexp = exp.diff_abundance(shuffler=_pair_shuffler, field=field, val1=val1, val2=val2, random_seed=random_seed, transform=transform, **kwargs)
    return newexp


def _new_experiment_from_pvals(cexp, exp, keep, odif, pvals, qvals):
    '''Combine the pvalues and effect size into a new experiment.

    Keep only the significant features, sort the features by the effect size

    Parameters
    ----------
    cexp : Experiment
        The experiment used for the actual diff. abundance (filtered for relevant samples/non-zero features)
    exp : Experiment or None
        The original experiment being analysed (with all samples/features)
        if None, keep only the samples from cexp
    keep : np.array of bool
        One entry per exp feature. True for the features which are significant (following FDR correction)
    odif : np.array of float
        One entry per exp feature. The effect size per feature (can be positive or negative)
    pvals : np.array of float
        One entry per exp feature. The p-value associated with each feature (not FDR corrected)
    qvals: np.array of float
        One entry per exp feature. The q-value associated with each feature (FDR corrected)

    Returns
    -------
    Experiment
        Containing only significant features, sorted by effect size.
        Each feature contains 2 new metadata fields: "_calour_pval", "_calour_stat"
    '''
    keep = np.where(keep)
    if len(keep[0]) == 0:
        logger.warning('no significant features found')
    newexp = cexp.reorder(keep[0], axis=1)
    if exp is not None:
        # we want all samples (rather than the subset in cexp) so use the original exp
        newexp = exp.filter_ids(newexp.feature_metadata.index.values)
    odif = odif[keep[0]]
    pvals = pvals[keep[0]]
    qvals = qvals[keep[0]]

    # this is for the p-value sorting in reverse order
    # it is fixed after the p-value sorting
    pvals[odif > 0] *= -1

    newexp.feature_metadata[_CALOUR_STAT] = odif
    newexp.feature_metadata[_CALOUR_PVAL] = pvals
    newexp.feature_metadata[_CALOUR_QVAL] = qvals

    # first sort by p-value (so within same effect size will be sorted)
    newexp.sort_by_metadata(_CALOUR_PVAL, axis='f', inplace=True)
    # fix the negative p-values used for the sort
    newexp.feature_metadata[_CALOUR_PVAL] = np.abs(newexp.feature_metadata[_CALOUR_PVAL])
    # now sort by effect size
    newexp.sort_by_metadata(_CALOUR_STAT, axis='f', inplace=True)

    return newexp
