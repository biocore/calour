'''
filtering (:mod:`calour.filtering`)
===================================

.. currentmodule:: calour.filtering

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   filter_by_data
   filter_by_metadata
   filter_samples
   filter_ids
   filter_mean
   filter_prevalence
   filter_min_abundance
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from heapq import nlargest
from logging import getLogger
from collections import Callable

import numpy as np

from .experiment import Experiment


logger = getLogger(__name__)


@Experiment._record_sig
def downsample(exp, field, axis=0, num_keep=None, inplace=False):
    '''Downsample the data set.

    This down samples all the samples/features to have the same number of
    samples/features for each categorical value of the field in
    ``sample_metadata`` or ``feature_metadata``.

    Parameters
    ----------
    field : str
        The name of the column in samples metadata table. This column
        should has categorical values
    axis : 0 / 1 (optional)
        0 (default) to filter samples, 1 to filter features
    num_keep : int or None (optional)
        None (default) to downsample to minimal group size.
        int : downsample to num_keep samples/features per group, drop values
        with < num_keep
    inplace : bool (optional)
        False (default) to do the filtering on a copy.
        True to do the filtering on the original ``Experiment``

    Returns
    -------
    ``Experiment``
    '''
    logger.debug('downsample on field %s' % field)
    if axis == 0:
        x = exp.sample_metadata
    elif axis == 1:
        x = exp.feature_metadata
    # convert to string type because nan values, if they exist in the column,
    # will fail `np.unique`
    values = x[field].astype(str).values
    unique, counts = np.unique(values, return_counts=True)
    if num_keep is None:
        num_keep = counts.min()
    indices = []
    num_skipped = 0
    for i in unique:
        i_indice = np.where(values == i)[0]
        if len(i_indice) < num_keep:
            num_skipped += 1
            continue
        elif len(i_indice) == num_keep:
            indices.append(i_indice)
        else:
            indices.append(np.random.choice(i_indice, num_keep))
    if num_skipped > 0:
        logger.info('%d values had < %d items and were skipped' % (num_skipped, num_keep))
    # if nothing left, raise error
    if len(indices) == 0:
        raise ValueError('No groups have more than %d items' % num_keep)
    return exp.reorder(np.concatenate(indices), axis=axis, inplace=inplace)


@Experiment._record_sig
@Experiment._convert_axis_name
def filter_by_metadata(exp, field, pick, axis=0, negate=False, inplace=False):
    '''Filter samples or features by metadata.

    Parameters
    ----------
    field : str
        the column name of the sample or feature metadata tables
    pick : list, tuple, or Callable
        pick what to keep based on the value in the specified field
    axis : 0 or 1, optional
        the field is on samples (0) or features (1) metadata
    negate : bool, optional
        discard instead of keep the pick if set to ``True``
    inplace : bool, optional
        do the filtering on the original ``Experiment`` object or a copied one.

    Returns
    -------
    ``Experiment``
        the filtered object
    '''
    logger.debug('filter_by_metadata')

    if axis == 0:
        x = exp.sample_metadata
    elif axis == 1:
        x = exp.feature_metadata
    else:
        raise ValueError('unknown axis %s' % axis)

    if isinstance(pick, Callable):
        select = pick(x[field])
    else:
        if not isinstance(pick, (list, tuple)):
            pick = [pick]

        select = x[field].isin(pick).values

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
        Apply predicate on each row (samples) (0) or each column (features) (1)
    negate : bool
        negate the predicate for selection
    kwargs : dict
        keyword argument passing to predicate function

    Returns
    -------
    ``Experiment``
        the filtered object
    '''
    func = {'sum_abundance': _sum_abundance,
            'freq_ratio': _freq_ratio,
            'unique_cut': _unique_cut,
            'mean_abundance': _mean_abundance,
            'prevalence': _prevalence}
    if isinstance(predicate, str):
        predicate = func[predicate]

    logger.debug('filter_by_data using function %r' % predicate)

    if exp.sparse:
        n = exp.data.shape[axis]
        select = np.ones(n, dtype=bool)
        if axis == 0:
            for row in range(n):
                # convert the row from sparse to dense, and cast to 1d array
                select[row] = predicate(exp.data[row, :].todense().A1, **kwargs)
        elif axis == 1:
            for col in range(n):
                # convert the column from sparse to dense, and cast to 1d array
                select[col] = predicate(exp.data[:, col].todense().A1, **kwargs)
        else:
            raise ValueError('unknown axis %s' % axis)
    else:
        select = np.apply_along_axis(predicate, 1 - axis, exp.data, **kwargs)

    if negate is True:
        select = ~ select

    logger.info('%s remaining' % np.sum(select))
    return exp.reorder(select, axis=axis, inplace=inplace)


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
    unique, counts = np.unique(np.array(x), return_counts=True)
    max_1, max_2 = nlargest(2, counts)
    return max_1 / max_2 <= ratio


@Experiment._record_sig
def filter_samples(exp, field, values, negate=False, inplace=False):
    '''Shortcut for filtering samples.'''
    return filter_by_metadata(exp, field=field, pick=values,
                              negate=negate, inplace=inplace)


@Experiment._record_sig
def filter_min_abundance(exp, min_abundance, **kwargs):
    '''Filter keeping only features with >= min_abundance total over all samples
    This is a convenience function wrapping filter_by_data()

    Parameters
    ----------
    min_abundance : numeric
        The minimal total abundance for each feature over all samples
    '''
    newexp = exp.filter_by_data('sum_abundance', axis=1, cutoff=min_abundance, **kwargs)
    return newexp


@Experiment._record_sig
def filter_prevalence(exp, fraction=0.5, cutoff=1/10000, **kwargs):
    '''Filter features keeping only ones present in at least fraction fraction of the samples.
    This is a convenience function wrapping filter_by_data()

    Parameters
    ----------
    fraction : float (optional)
        Keep features present at least in fraction of samples
    cutoff : float (optional)
        The minimal fraction of reads for the otu to be called present in a sample

    Returns
    -------
    ``Experiment`` with only features present in at least fraction of samples
    '''
    newexp = exp.filter_by_data('prevalence', axis=1, fraction=fraction, cutoff=cutoff, **kwargs)
    return newexp


def filter_mean(exp, cutoff=0.01, **kwargs):
    '''Filter features with a mean at least cutoff of the mean total abundance/sample

    In order to keep features with mean abundance of 1%, use ``filter_mean(cutoff=0.01)``

    Parameters
    ----------
    cutoff : float (optional)
        The minimal mean abundance fraction (out of the mean of total abundance per sample) for a feature in order
        to keep it. Default is 0.01 - keep features with mean abundance >=1% of mean total abundance per sample

    Returns
    -------
    ``Experiment``
    '''
    factor = np.mean(exp.data.sum(axis=1))
    newexp = exp.filter_by_data('mean_abundance', axis=1, cutoff=cutoff * factor, **kwargs)
    return newexp


def filter_ids(exp, ids, axis=1, negate=False, inplace=False):
    '''Filter samples or features based on a list index values

    Parameters
    ----------
    ids : iterable of str
        the feature/sample ids to filter (index values)
    axis : int (optional)
        1 (default) to filter features, 0 to filter samples
    negate : bool (optional)
        negate the filtering
    inplace : bool (optional)
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    ``Experiment``
        filtered so contains only features/samples present in exp and in ids
    '''
    logger.debug('filter_ids')
    okpos = []
    tot_ids = 0
    if axis == 0:
        index = exp.sample_metadata.index
    else:
        index = exp.feature_metadata.index
    for cid in ids:
        tot_ids += 1
        if cid in index:
            pos = index.get_loc(cid)
            okpos.append(pos)
    logger.debug('list contained %d sequences. Found %d sequences in experiment' % (tot_ids, len(okpos)))
    if negate:
        okpos = np.setdiff1d(np.arange(len(index)), okpos, assume_unique=True)

    newexp = exp.reorder(okpos, axis=axis, inplace=inplace)
    return newexp
