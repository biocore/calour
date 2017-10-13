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
from sklearn.preprocessing import LabelEncoder

from .experiment import Experiment
from .util import _to_list
from . import dsfdr


logger = getLogger(__name__)


@Experiment._record_sig
def correlation(exp: Experiment, field, method='spearman', nonzero=False, transform=None, numperm=1000, alpha=0.1, fdr_method='dsfdr'):
    '''Find features with correlation to a numeric metadata field

    With permutation based p-values and multiple hypothesis correction

    Parameters
    ----------
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
    fdr_method : str
        method to compute FDR. Allowed method include "", ""

    Returns
    -------
    :class:`.Experiment`
        The experiment with only correlated features, sorted according to correlation coefficient
    '''
    cexp = exp.filter_abundance(0, strict=True)

    data = cexp.get_data(copy=True, sparse=False).transpose()

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
    return _new_experiment_from_pvals(cexp, exp, keep, odif, pvals)


@Experiment._record_sig
def diff_abundance(exp: Experiment, field, *args, method='delta mean', fdr_method='dsfdr',
                   transform='rankdata', numperm=1000, alpha=0.1):
    '''Test the differential abundance between groups.

    Using permutation based statistic tests to identify features that
    are differentially abundant in sample groups.

    Parameters
    ----------
    field : str
        The field of metadata to group samples
    args : tuple of str or list of str
        These give the sample groups.
    method : str or callable
        The statistic method to test difference. It can be a callable that takes
        input of data, labels and outputs array of float). The builtin options:

        * 'mw', 'mann-whitney' or 'wilcoxon rank-sum' : Mann-Whitney U
          test (aka Wilcoxon rank-sum test). This is a nonparametric
          test between 2 independent sample groups

        * 'kw' or 'kruskal-wallis' : Kruskal-Wallis test. This is an
          extension of Mann-Whitney U test to more than 2 sample
          groups.

        * 'delta mean' : compute the delta between the means of 2
          sample groups (i.e. mean(group 1) - mean(group 2)) and
          compare it to the random permutations.

        * 'delta mean norm' : compute the delta means normalized by
          standard deviations between 2 sample groups
          (i.e. (mean(group 1) - mean(group 2)) / (std(group 1) +
          std(group 2))) and compare it to the random permutations.

    fdr_method : str
        FDR method. Choice include 'bonferroni', 'dsFDR', 'BH', ''
    transform : str or None
        transformation to apply to the data before caluculating the statistic
        'rankdata' : rank transfrom each OTU reads
        'log2data' : calculate log2 for each OTU using minimal cutoff of 2
        'normdata' : normalize the data to constant sum per samples
        'binarydata' : convert to binary absence/presence
    alpha : float
        the desired FDR false discovery level
    numperm : int
        number of permutations to perform

    Returns
    -------
    :class:`.Experiment`
        The experiment with only significant (FDR<=maxfval) difference, sorted according to difference

    Examples
    --------

    Let's say you have 4 sample types ('HC', 'CD', 'UC', and 'IBS')
    labeled in the column of 'sample_type' in the sample metadata. To
    identify features that are different between 'HC' and 'CD' sample
    groups:

    >>> exp.diff_abundance('sample_type', 'HC', 'CD')   # doctest: +SKIP

    To identify features that are different between 'HC' group and 'CD' and 'UC' groups combined:

    >>> exp.diff_abundance('sample_type', 'HC', ['CD', 'UC'])   # doctest: +SKIP

    To identify features that are different between 'HC', 'IBS' and 'CD', 'UC':

    >>> exp.diff_abundance('sample_type', ['HC', 'IBS'], ['CD', 'UC'])   # doctest: +SKIP

    To identify features that are different between any pair of the 4 groups:

    >>> exp.diff_abundance('sample_type')   # doctest: +SKIP

    '''
    logger.info('The groups for diff abundance test:\n%r' % exp.sample_metadata[field].value_counts())
    # remove features not present in any group
    exp = exp.filter_abundance(0, strict=True)
    data = exp.get_data(copy=True, sparse=False).transpose()
    # prepare the labels.
    le = LabelEncoder()
    # transform the str labels into int labels
    labels = le.fit_transform(exp.sample_metadata[field])
    keep, odif, pvals = dsfdr.dsfdr(data, labels, method=method, transform_type=transform, alpha=alpha, numperm=numperm, fdr_method=fdr_method)
    logger.info('method %s. number of higher in %s : %d. number of higher in %s : %d. total %d' % (method, val1, np.sum(odif[keep] > 0), val2, np.sum(odif[keep] < 0), np.sum(keep)))
    return _new_experiment_from_pvals(exp, exp, keep, odif, pvals)


def _new_experiment_from_pvals(cexp, exp, keep, odif, pvals):
    '''Combine the pvalues and effect size into a new experiment.
    Keep only the significant features, sort the features by the effect size

    Parameters
    ----------
    cexp : :class:`.Experiment`
        The experiment used for the actual diff. abundance (filtered for relevant samples/non-zero features)
    exp : :class:`.Experiment`
        The original experiment being analysed (with all samples/features)
    keep : np.array of bool
        One entry per exp feature. True for the features which are significant (following FDR correction)
    odif : np.array of float
        One entry per exp feature. The effect size per feature (can be positive or negative)
    pval : np.array of float
        One entry per exp feature. The p-value associated with each feature (not FDR corrected)

    Returns
    -------
    :class:`.Experiment`
    Containing only significant features, sorted by effect size.
    Each feature contains 2 new metadata fields: _calour_diff_abundance_pval, _calour_diff_abundance_effect
    '''
    keep = np.where(keep)
    if len(keep[0]) == 0:
        logger.warn('no significant features found')
    newexp = cexp.reorder(keep[0], axis=1)
    if exp is not None:
        # we want all samples (rather than the subset in cexp) so use the original exp
        newexp = exp.filter_ids(newexp.feature_metadata.index.values)
    odif = odif[keep[0]]
    pvals = pvals[keep[0]]
    si = np.argsort(odif, kind='mergesort')
    odif = odif[si]
    pvals = pvals[si]
    newexp = newexp.reorder(si, axis=1)
    newexp.feature_metadata['_calour_diff_abundance_effect'] = odif
    newexp.feature_metadata['_calour_diff_abundance_pval'] = pvals
    return newexp
