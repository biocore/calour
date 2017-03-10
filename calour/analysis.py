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
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

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


def get_term_features(seqs, sequence_annotations):
    '''Get dict of number of appearances in each sequence keyed by term

    Parameters
    ----------
    seqs : list of str
        A list of DNA sequences
    sequence_annotations : dict of (sequence, list of ontology terms)
        from dbbact.get_seq_list_fast_annotations()

    Returns
    -------
    seq_annotations : dict of (ontology_term : num per sequence)
        number of times each ontology term appears in each sequence in seqs
    '''
    # get all terms
    terms = set()
    for ctermlist in sequence_annotations.values():
        for cterm in ctermlist:
            terms.add(cterm)

    seq_annotations = {}
    for cterm in terms:
        seq_annotations[cterm] = np.zeros([len(seqs)])

    num_seqs_no_annotations = 0
    for idx, cseq in enumerate(seqs):
        if cseq not in sequence_annotations:
            num_seqs_no_annotations += 1
            continue
        for cterm in sequence_annotations[cseq]:
            seq_annotations[cterm][idx] += 1
    if num_seqs_no_annotations > 0:
        logger.info('found %d sequences with no annotations out of %d' % (num_seqs_no_annotations, len(seqs)))
    return seq_annotations


def relative_enrichment(exp, features, feature_terms):
    '''Get the list of enriched terms in features compared to all features in exp.

    given uneven distribtion of number of terms per feature

    Parameters
    ----------
    exp : calour.Experiment
        The experiment to compare the features to
    features : list of str
        The features (from exp) to test for enrichmnt
    feature_terms : dict of {feature: list of terms}
        The terms associated with each feature in exp
        feature (key) : str the feature (out of exp) to which the terms relate
        feature_terms (value) : list of str or int the terms associated with this feature

    Returns
    -------
    list of dict
        info about significantly enriched terms. one item per term, keys are:
        'pval' : the p-value for the enrichment (float)
        'observed' : the number of observations of this term in group1 (int)
        'expected' : the expected (based on all features) number of observations of this term in group1 (float)
        'group1' : fraction of total terms in group 1 which are the specific term (float)
        'group2' : fraction of total terms in group 2 which are the specific term (float)
        'description' : the term (str)
    '''
    all_features = set(exp.feature_metadata.index.values)
    bg_features = list(all_features.difference(features))
    # get the number of features each term appears in the bg and fg feature lists
    bg_terms = get_term_features(bg_features, feature_terms)
    fg_terms = get_term_features(features, feature_terms)

    for cterm in bg_terms.keys():
        bg_terms[cterm] = np.sum(bg_terms[cterm])
    for cterm in fg_terms.keys():
        fg_terms[cterm] = np.sum(fg_terms[cterm])

    total_bg_terms = np.sum(list(bg_terms.values()))
    total_fg_terms = np.sum(list(fg_terms.values()))
    total_terms = total_bg_terms + total_fg_terms

    # calculate total count for each feature in both lists combined
    all_terms = bg_terms.copy()
    for cterm, ccount in fg_terms.items():
        if cterm not in all_terms:
            all_terms[cterm] = 0
        all_terms[cterm] += fg_terms[cterm]

    allp = []
    pv = []
    for cterm in all_terms.keys():
        pval = all_terms[cterm] / total_terms
        num1 = fg_terms.get(cterm, 0)
        num2 = bg_terms.get(cterm, 0)
        pval1 = 1 - stats.binom.cdf(num1, total_fg_terms, pval)
        pval2 = 1 - stats.binom.cdf(num2, total_bg_terms, pval)
        p = np.min([pval1, pval2])
        # store the result
        allp.append(p)
        cpv = {}
        cpv['pval'] = p
        cpv['observed'] = fg_terms[cterm]
        cpv['expected'] = total_fg_terms * pval
        cpv['group1'] = num1 / total_fg_terms
        cpv['group2'] = num2 / total_bg_terms
        cpv['description'] = cterm
        pv.append(cpv)

    reject, _, _, _ = multipletests(allp, method='fdr_bh')
    keep = np.where(reject)[0]
    plist = []
    rat = []
    for cidx in keep:
        plist.append(pv[cidx])
        rat.append(np.abs(float(pv[cidx]['observed'] - pv[cidx]['expected'])) / np.mean([pv[cidx]['observed'], pv[cidx]['expected']]))
    print('found %d' % len(keep))
    si = np.argsort(rat)
    si = si[::-1]
    newplist = []
    for idx, crat in enumerate(rat):
        newplist.append(plist[si[idx]])

    return(newplist)
