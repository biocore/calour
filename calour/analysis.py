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

from .experiment import Experiment
from .util import _to_list
from . import dsfdr


logger = getLogger(__name__)


@Experiment._record_sig
def correlation(exp, field, method='spearman', nonzero=False, transform='rankdata', numperm=1000, alpha=0.1, fdr_method='dsfdr'):
    '''Find features with correlation to a numeric metadata field
    With permutation based p-values and multiple hypothesis correction

    Parameters
    ----------
    exp: calour.Experiment
    field: str
        The field to test by. Values are converted to numeric.
    method : str or function
        the method to use for the t-statistic test. options:
        'spearman' : spearman correlation (numeric)
        'pearson' : pearson correlation (numeric)
        function : use this function to calculate the t-statistic (input is data,labels, output is array of float)
    nonzero : bool (optional)
        True to calculate the correlation only for samples where the feature is present (>0).
        False (default) to calculate the correlation over all samples
        Note: setting nonzero to True slows down the calculation
        Note: can be set to True only using 'spearman' or 'pearson', not using a custom function
    transform : str or None
        transformation to apply to the data before caluculating the statistic
        'rankdata' : rank transfrom each OTU reads
        'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        'normdata' : normalize the data to constant sum per samples
        'binarydata' : convert to binary absence/presence
    alpha : float
        the desired FDR control level
    numperm : int
        number of permutations to perform

    Returns
    -------
    newexp : calour.Experiment
        The experiment with only significant (FDR<=maxfval) correlated features, sorted according to correlation size
    '''
    data = exp.get_data(copy=True, sparse=False).transpose()
    labels = pd.to_numeric(exp.sample_metadata[field], errors='coerce').values
    # remove the nans
    nanpos = np.where(np.isnan(labels))[0]
    if len(nanpos) > 0:
        logger.warn('NaN values encountered in labels for correlation. Ignoring these samples')
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
    keep, odif, pvals = dsfdr.dsfdr(data, labels, method=method, transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method)
    logger.info('method %s for field %s. Positive correlated features : %d. Negative correlated features : %d. total %d'
                % (method, field, np.sum(odif[keep] > 0), np.sum(odif[keep] < 0), np.sum(keep)))
    return _new_experiment_from_pvals(exp, keep, odif, pvals)


@Experiment._record_sig
def diff_abundance(exp, field, val1, val2=None, method='meandiff', transform='rankdata', numperm=1000, alpha=0.1, fdr_method='dsfdr'):
    '''
    test the differential expression between 2 groups (val1 and val2 in field field)
    using permutation based fdr (dsfdr)
    for bacteria that have a significant difference.

    Parameters
    ----------
    exp: calour.Experiment
    field: str
        The field to test by
    val1: str or list of str
        The values for the first group.
    val1: str or list of str or None (optional)
        None (default) to compare to all other samples (not in val1)
    method : str or function
        the method to use for the t-statistic test. options:
        'meandiff' : mean(A)-mean(B) (binary)
        'mannwhitney' : mann-whitneu u-test (binary)
        'kruwallis' : kruskal-wallis test (multiple groups)
        'stdmeandiff' : (mean(A)-mean(B))/(std(A)+std(B)) (binary)
        function : use this function to calculate the t-statistic (input is data,labels, output is array of float)
    transform : str or None
        transformation to apply to the data before caluculating the statistic
        'rankdata' : rank transfrom each OTU reads
        'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        'normdata' : normalize the data to constant sum per samples
        'binarydata' : convert to binary absence/presence
    alpha : float
        the desired FDR control level
    numperm : int
        number of permutations to perform

    Returns
    -------
    newexp : calour.Experiment
        The experiment with only significant (FDR<=maxfval) difference, sorted according to difference
    '''

    # if val2 is not none, need to get rid of all other samples (not val1/val2)
    val1 = _to_list(val1)
    if val2 is not None:
        val2 = _to_list(val2)
        cexp = exp.filter_samples(field, val1+val2, negate=False)
        logger.info('%d samples with both values' % cexp.shape[0])
    else:
        cexp = exp

    data = cexp.get_data(copy=True, sparse=False).transpose()
    # prepare the labels.
    labels = np.zeros(len(cexp.sample_metadata))
    labels[cexp.sample_metadata[field].isin(val1).values] = 1
    logger.info('%d samples with value 1 (%s)' % (np.sum(labels), val1))
    keep, odif, pvals = dsfdr.dsfdr(data, labels, method=method, transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method)
    logger.info('method %s. number of higher in %s : %d. number of higher in %s : %d. total %d' % (method, val1, np.sum(odif[keep] > 0), val2, np.sum(odif[keep] < 0), np.sum(keep)))
    return _new_experiment_from_pvals(exp, keep, odif, pvals)


@Experiment._record_sig
def diff_abundance_kw(exp, field, transform='rankdata', numperm=1000, alpha=0.1, fdr_method='dsfdr'):
    '''Test the differential expression between multiple sample groups using the Kruskal Wallis test.
    uses a permutation based fdr (dsfdr) for bacteria that have a significant difference.

    Parameters
    ----------
    exp: calour.Experiment
    field: str
        The field to test by
    transform : str or None
        transformation to apply to the data before caluculating the statistic
        'rankdata' : rank transfrom each OTU reads
        'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        'normdata' : normalize the data to constant sum per samples
        'binarydata' : convert to binary absence/presence
    alpha : float
        the desired FDR control level
    numperm : int
        number of permutations to perform

    Returns
    -------
    newexp : calour.Experiment
        The experiment with only significant (FDR<=maxfval) difference, sorted according to difference
    '''
    logger.debug('diff_abundance_kw for field %s' % field)
    data = exp.get_data(copy=True, sparse=False).transpose()
    # prepare the labels. If correlation method, get the values, otherwise the group
    labels = np.zeros(len(exp.sample_metadata))
    for idx, clabel in enumerate(exp.sample_metadata[field].unique()):
        labels[exp.sample_metadata[field].values == clabel] = idx
    logger.debug('Found %d unique sample labels' % (idx+1))
    keep, odif, pvals = dsfdr.dsfdr(data, labels, method='kruwallis', transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method)
    print(keep)
    logger.info('Found %d significant features' % (np.sum(keep)))
    return _new_experiment_from_pvals(exp, keep, odif, pvals)


@Experiment._record_sig
def _new_experiment_from_pvals(exp, keep, odif, pvals):
    '''Combine the pvalues and effect size into a new experiment.
    Keep only the significant features, sort the features by the effect size

    Parameters
    ----------
    exp : ``Experiment``
        The experiment being analysed
    keep : np.array of bool
        One entry per exp feature. True for the features which are significant (following FDR correction)
    odif : np.array of float
        One entry per exp feature. The effect size per feature (can be positive or negative)
    pval : np.array of float
        One entry per exp feature. The p-value associated with each feature (not FDR corrected)

    Returns
    -------
    ``Experiment``
    Containing only significant features, sorted by effect size.
    Each feature contains 2 new metadata fields: _calour_diff_abundance_pval, _calour_diff_abundance_effect
    '''
    keep = np.where(keep)
    if len(keep[0]) == 0:
        logger.warn('no significant features found')
    newexp = exp.reorder(keep[0], axis=1)
    odif = odif[keep[0]]
    pvals = pvals[keep[0]]
    si = np.argsort(odif, kind='mergesort')
    odif = odif[si]
    pvals = pvals[si]
    newexp = newexp.reorder(si, axis=1)
    newexp.feature_metadata['_calour_diff_abundance_effect'] = odif
    newexp.feature_metadata['_calour_diff_abundance_pval'] = pvals
    return newexp
