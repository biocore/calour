# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from heapq import nlargest
from logging import getLogger

import scipy
import numpy as np

from .experiment import Experiment


logger = getLogger(__name__)


@Experiment._record_sig
def down_sample(exp, field, axis=0, inplace=False):
    '''Down sample the data set.

    This down samples all the samples to have the same number of
    samples for each categorical value of the field in
    ``sample_metadata`` or ``feature_metadata``.

    Parameters
    ----------
    field : str
        The name of the column in samples metadata table. This column
        should has categorical values

    Returns
    -------
    Experiment
    '''
    if axis == 0:
        x = exp.sample_metadata
    elif axis == 1:
        x = exp.feature_metadata
    values = x[field].values
    unique, counts = np.unique(values, return_counts=True)
    min_index = counts.argmin()
    min_value = unique[min_index]
    min_count = counts[min_index]
    indices = []
    for i in unique:
        i_indice = np.where(values == i)[0]
        if i == min_value:
            indices.append(i_indice)
        else:
            indices.append(np.random.choice(i_indice, min_count))
    return exp.reorder(np.concatenate(indices), axis=axis, inplace=inplace)


@Experiment._record_sig
def filter_by_metadata(exp, field, values, axis=0, negate=False, inplace=False):
    '''Filter samples or features by metadata.

    Parameters
    ----------
    field : str
        the column name or sample or feature metadata
    values : list, tuple, or numeric/str
    axis : 0 or 1
        the column name is on samples (0) or features (1) metadata
    '''
    logger.debug('filter_by_metadata')

    if not isinstance(values, (list, tuple)):
        values = [values]

    if axis == 0:
        x = exp.sample_metadata
    elif axis == 1:
        x = exp.feature_metadata

    select = x[field].isin(values).values
    if negate is True:
        select = ~ select
    return exp.reorder(select, axis=axis, inplace=inplace)


@Experiment._record_sig
def filter_by_data(exp, predicate, axis=0, negate=False, inplace=False, **kwargs):
    '''Filter samples or features by data.

    Parameters
    ----------
    predicate : str or callable
        The callable accepts a list of numeric and return a bool. Alternatively
        it also accepts the following strings:
        'sum_abundance': calls ``_sum_abundance``,
        'freq_ratio': calls ``_freq_ratio``,
        'unique_cut': calls ``_unique_cut``,
        'mean_abundance': calls ``_mean_abundance``,
        'prevalence': calls ``_prevalence``
    axis : 0 or 1
        Apply predicate on row (samples) (0) or column (features) (1)
    negate : bool
        negate the predicate for selection
    kwargs : dict
        keyword argument passing to predicate function

    Returns
    -------
    exp : Experiment

    '''
    select = _filter_by_data(exp.data, predicate, axis, negate, **kwargs)
    logger.info('%s remaining' % np.sum(select))
    return exp.reorder(select, axis=axis, inplace=inplace)


def _filter_by_data(data, predicate, axis=0, negate=False, **kwargs):
    func = {'sum_abundance': _sum_abundance,
            'freq_ratio': _freq_ratio,
            'unique_cut': _unique_cut,
            'mean_abundance': _mean_abundance,
            'prevalence': _prevalence}
    if isinstance(predicate, str):
        predicate = func[predicate]

    if scipy.sparse.issparse(data):
        n = data.shape[axis]
        select = np.ones(n, dtype=bool)
        if axis == 0:
            for row in range(n):
                # convert the row from sparse to dense, and cast to 1d array
                select[row] = predicate(data[row, :].todense().A1, **kwargs)
        elif axis == 1:
            for col in range(n):
                # convert the column from sparse to dense, and cast to 1d array
                select[col] = predicate(data[:, col].todense().A1, **kwargs)
    else:
        select = np.apply_along_axis(predicate, 1 - axis, data, **kwargs)

    if negate is True:
        select = ~ select

    return select


def _sum_abundance(x, cutoff=10):
    '''Check if the sum abundance larger than cutoff.

    It can be used filter features with at least "cutoff" abundance
    total over all samples

    Examples
    --------
    >>> _sum_abundance(np.array([0, 1, 1]), 2)
    True
    >>> _sum_abundance(np.array([0, 1, 1]), 2.01)
    False

    '''
    logger.debug('')
    return x.sum() >= cutoff


def _mean_abundance(x, cutoff=0.01):
    '''Check if the mean abundance larger than cutoff.

    Can be used to keep features with means at least "cutoff" in all
    samples

    Examples
    --------
    >>> _mean_abundance(np.array([0, 0, 1, 1]), 0.51)
    False
    >>> _mean_abundance(np.array([0, 0, 1, 1]), 0.5)
    True

    '''
    logger.debug('')
    return x.mean() >= cutoff


def _prevalence(x, cutoff=1/10000, fraction=0.5):
    '''Check the prevalence of values above the cutoff.

    present (abundance >= cutoff) in at least "fraction" of samples

    Examples
    --------
    >>> _prevalence(np.array([0, 1]))
    True
    >>> _prevalence(np.array([0, 1, 2, 3]), 2, 0.5)
    True
    >>> _prevalence(np.array([0, 1, 2]), 2, 0.51)
    False
    '''
    logger.debug('')
    frac = np.sum(x >= cutoff) / len(x)
    return frac >= fraction


def _unique_cut(x, unique=0.05):
    '''the percentage of distinct values out of the number of total samples.

    Examples
    --------
    >>> _unique_cut([0, 0], 0.49)
    True
    >>> _unique_cut([0, 0], 0.51)
    False
    >>> _unique_cut([0, 1], 1.01)
    False
    '''
    logger.debug('')
    count = len(set(x))
    return count / len(x) >= unique


def _freq_ratio(x, ratio=2):
    '''the ratio of the most common value to the second most common value

    Return True if the ratio is not greater than "ratio".

    Examples
    --------
    >>> _freq_ratio([0, 0, 1, 2], 2)
    True
    >>> _freq_ratio([0, 0, 1, 1], 1.01)
    True
    >>> _freq_ratio([0, 0, 1, 2], 1.99)
    False
    '''
    logger.debug('')
    unique, counts = np.unique(np.array(x), return_counts=True)
    max_1, max_2 = nlargest(2, counts)
    return max_1 / max_2 <= ratio


def filter_samples(exp, field, values, negate=False, inplace=False):
    '''Shortcut for filtering samples.'''
    return filter_by_metadata(exp, field=field, values=values,
                              negate=negate, inplace=inplace)


def filter_taxonomy(exp, values, negate=False, inplace=False, substring=True):
    '''filter keeping only observations with taxonomy string matching taxonomy

    if substring=True, look for partial match instead of identity
    '''
    if 'taxonomy' not in exp.feature_metadata.columns:
        logger.warn('No taxonomy field in experiment')
        return None

    if not isinstance(values, (list, tuple)):
        values = [values]

    taxstr = [';'.join(x).lower() for x in exp.feature_metadata['taxonomy']]

    select = np.zeros(len(taxstr), dtype=bool)
    for cval in values:
        if substring:
            select += [cval.lower() in ctax for ctax in taxstr]
        else:
            select += [cval.lower() == ctax for ctax in taxstr]

    if negate is True:
        select = ~ select

    logger.warn('%s remaining' % np.sum(select))
    return exp.reorder(select, axis=1, inplace=inplace)
