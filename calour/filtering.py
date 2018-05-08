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
   filter_prevalence
   filter_abundance
   filter_mean_abundance
   filter_sample_categories
   downsample
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
import reprlib

import numpy as np
from scipy.sparse import issparse

from . import Experiment
from ._doc import ds
from .util import _to_list


logger = getLogger(__name__)


@Experiment._record_sig
def downsample(exp: Experiment, field, axis=0, num_keep=None, inplace=False):
    '''Downsample the data set.

    This down samples all the samples/features to have the same number of
    samples/features for each categorical value of the field in
    `sample_metadata` or `feature_metadata`.

    Parameters
    ----------
    field : str
        The name of the column in samples metadata table. This column
        should has categorical values
    axis : 0, 1, 's', or 'f', optional
        0 or 's' (default) to filter samples; 1 or 'f' to filter features
    num_keep : int or None, optional
        None (default) to downsample to minimal group size.
        int : downsample to num_keep samples/features per group, drop values
        with < num_keep
    inplace : bool, optional
        False (default) to do the filtering on a copy.
        True to do the filtering on the original :class:`.Experiment`

    Returns
    -------
    Experiment

    See Also
    --------
    filter_sample_categories
    '''
    logger.debug('downsample on field %s' % field)
    if axis == 0:
        x = exp.sample_metadata
        error_axis_name = 'sample'
    elif axis == 1:
        x = exp.feature_metadata
        error_axis_name = 'feature'

    if field not in x:
        raise ValueError('Field %s not in %s_metadata. (fields are: %s)' % (field, error_axis_name, x.columns))

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
            indices.append(np.random.choice(i_indice, num_keep, replace=False))
    if num_skipped > 0:
        logger.info('%d values had < %d items and were skipped' % (num_skipped, num_keep))
    # if nothing left, raise error
    if len(indices) == 0:
        raise ValueError('No groups have more than %d items' % num_keep)
    return exp.reorder(np.concatenate(indices), axis=axis, inplace=inplace)


@Experiment._record_sig
def filter_sample_categories(exp: Experiment, field, min_samples=5, inplace=False):
    '''Filter sample categories that have too few samples.

    This is useful to get rid of categories with few samples for
    supervised classification training.  It also drops the samples
    that don't have any value in the field.

    Examples
    --------

    Parameters
    ----------
    field : str
        The name of the column in samples metadata table. This column
        should has categorical values
    min_samples : int, optional
        Filter away the samples with a value in the given column if its sample count is
        less than min_samples.
    inplace : bool, optional
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    Experiment

    See Also
    --------
    downsample
    '''
    if field not in exp.sample_metadata:
        raise ValueError('field %s not in sample_metadata (fields are: %s)' % (field, exp.sample_metadata.columns))
    exp = exp.reorder(exp.sample_metadata[field].notnull(), inplace=inplace)
    unique, counts = np.unique(exp.sample_metadata[field].values, return_counts=True)
    drop_values = [i for i, j in zip(unique, counts) if j < min_samples]
    if drop_values:
        logger.debug('Drop samples with {0} values in column {1}'.format(drop_values, field))
        return exp.filter_samples(field, drop_values, negate=True, inplace=inplace)
    else:
        return exp


@Experiment._record_sig
def filter_by_metadata(exp: Experiment, field, select, axis=0, negate=False, inplace=False):
    '''Filter samples or features by metadata.

    Parameters
    ----------
    field : str
        the column name of the sample or feature metadata tables
    select : None, Callable, or list/set/tuple-like
        select what to keep based on the value in the specified field.
        if it is a callable, it accepts a 1D array and return a
        boolean array of the same length; if it is a list/set/tuple-like object,
        keep the samples with the values in the `field` column included
        in the `select`; if it is None, filter out the NA.
    axis : 0, 1, 's', or 'f', optional
        the field is on samples (0 or 's') or features (1 or 'f') metadata
    negate : bool, optional
        discard instead of keep the select if set to `True`
    inplace : bool, optional
        do the filtering on the original :class:`.Experiment` object or a copied one.

    Returns
    -------
    Experiment
        the filtered object
    '''
    if axis == 0:
        x = exp.sample_metadata
        error_axis_name = 'sample'
    elif axis == 1:
        x = exp.feature_metadata
        error_axis_name = 'feature'
    else:
        raise ValueError('unknown axis %s' % axis)

    if field not in x:
        raise ValueError('Field %s not in %s_metadata. (fields are: %s)' % (field, error_axis_name, x.columns))

    if isinstance(select, Callable):
        select = select(x[field])
    elif select is None:
        select = x[field].notnull()
    else:
        select = x[field].isin(select).values

    if negate is True:
        select = ~ select
    return exp.reorder(select, axis=axis, inplace=inplace)


@ds.get_sectionsf('filtering.filter_by_data')
@Experiment._record_sig
def filter_by_data(exp: Experiment, predicate, axis=1, field=None, negate=False, inplace=False, **kwargs):
    '''Filter samples or features by data.

    Parameters
    ----------
    predicate : str or callable
        The callable accepts a list of numeric and return a bool. Alternatively
        it also accepts the following strings:

        * 'sum_abundance': calls :func:`.sum_abundance`,
        * 'mean_abundance': calls :func:`.mean_abundance`,
        * 'prevalence': calls :func:`.prevalence`
        * 'freq_ratio': calls :func:`.freq_ratio`,
        * 'unique_cut': calls :func:`.unique_cut`
    axis : 0, 1, 's', or 'f', optional
        Apply predicate on each row (ie samples) (0, 's') or each column (ie features) (1, 'f')
    field : str or `None`, optional
        The column in the sample_metadata (or feature_metadata,
        depending on `axis`). If it is `None`, the `predicate`
        operates on the whole data set; if it is not `None`, the data
        set is divided into groups according to the sample_metadata
        (feature_metadata) column and the `predicate` operates on each
        partition of data - only the features (or samples) that fail
        to pass every partition will be filtered away.
    negate : bool
        negate the predicate for selection
    kwargs : dict
        keyword argument passing to predicate function

    Returns
    -------
    Experiment
        the filtered object

    See Also
    --------
    filter_mean_abundance
    filter_abundance
    filter_prevalence

    '''
    if axis == 0:
        x = exp.feature_metadata
        data = exp.data.T
    elif axis == 1:
        x = exp.sample_metadata
        data = exp.data
    else:
        raise ValueError('unknown axis %s' % axis)

    if field is None:
        groups = [0]
        indices = np.zeros(data.shape[0])
    else:
        values = x[field].values.astype('U')
        groups, indices = np.unique(values, return_inverse=True)

    # functions that can be applied to full matrix
    # this is much faster
    func_vec = {'sum_abundance': sum_abundance,
                'mean_abundance': mean_abundance,
                'prevalence': prevalence}

    func_slow = {'freq_ratio': freq_ratio,
                 'unique_cut': unique_cut}
    logger.debug('filter_by_data using function %r' % predicate)

    n = data.shape[1]
    select = np.zeros(n, dtype='?')

    if predicate in func_vec:
        pred = func_vec[predicate]
    elif predicate in func_slow:
        pred = func_slow[predicate]
    else:
        pred = predicate

    for i, _ in enumerate(groups):
        if predicate in func_vec:
            select_i = pred(data[indices == i], axis=0, **kwargs)
        else:
            select_i = np.ones(n, dtype='?')
            for row in range(n):
                # convert the row from sparse to dense, and cast to 1d array
                select_i[row] = pred(data[row, indices == i].todense().A1, **kwargs)

        select = select | select_i

    if negate is True:
        select = ~ select

    logger.info('After filtering, %s remaining' % np.sum(select))
    return exp.reorder(select, axis=axis, inplace=inplace)


ds.keep_params('filtering.filter_by_data.parameters', 'negate')


def sum_abundance(data, axis, cutoff=10, strict=False):
    '''Check if the sum abundance larger than cutoff.

    It can be used filter features with at least "cutoff" abundance
    total over all samples.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
    axis : int
        0 to sum each column, 1 to sum each row. passed to :func:`numpy.sum`
    cutoff : float
        the mean threshold
    strict : bool, optional
        False (default) to use sum >= cutoff; True to use sum > cutoff

    Returns
    -------
    np.ndarray
        bool array with True if sum >= cutoff.

    Examples
    --------
    >>> np.sum(sum_abundance(np.array([[0, 1, 1]]), axis=1, cutoff=2)) == 1
    True
    >>> np.sum(sum_abundance(np.array([[0, 1, 1]]), axis=1, cutoff=2, strict=True)) == 0
    True
    >>> np.sum(sum_abundance(np.array([[0, 1, 1]]), axis=1, cutoff=2.01)) == 0
    True

    '''
    if strict:
        res = data.sum(axis=axis) > cutoff
    else:
        res = data.sum(axis=axis) >= cutoff
    if issparse(data):
        res = res.A1
    return res


def mean_abundance(data, axis, cutoff=0.01, strict=False):
    '''Check if the mean abundance larger than cutoff.

    Can be used to keep features with means at least "cutoff" in all
    samples

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
    axis : int
        0 to average each column, 1 to average each row. passed to :func:`numpy.mean`
    cutoff : float
        the mean threshold
    strict : bool, optional
        False (default) to use mean >= cutoff; True to use mean > cutoff

    Returns
    -------
    np.ndarray
        bool array with True if mean >= cutoff.

    Examples
    --------
    >>> mean_abundance(np.array([[0, 0, 1], [1, 1, 1]]), axis=1, cutoff=0.51).tolist()
    [False, True]
    >>> mean_abundance(np.array([0, 0, 1, 1]), axis=0, cutoff=0.5)
    True
    >>> mean_abundance(np.array([0, 0, 1, 1]), axis=0, cutoff=0.5, strict=True)
    False

    '''
    m = data.mean(axis=axis)
    if strict is True:
        res = m > cutoff
    else:
        res = m >= cutoff
    if issparse(data):
        res = res.A1
    return res


def prevalence(data, axis, cutoff=1/10000, fraction=0.1):
    '''Check the prevalence of values above the cutoff.

    present (abundance >= cutoff) in at least "fraction" of samples

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
    axis : int
        compute prevalence of each column (0) or row (1).
    cutoff : float
        the min threshold of abundance
    fraction : float
        [0, 1). the min threshold of presence (in fraction)

    Returns
    -------
    np.ndarray
        bool array with True if prevalence >= cutoff.

    Examples
    --------
    >>> x = prevalence(np.array([[0, 1, 2], [0, 1, 2]]), 0, 2, 0.51)
    >>> x.tolist()
    [False, False, True]
    >>> x = prevalence(np.array([[0, 1, 2], [0, 2, 2]]), 0, 2, 0.5)
    >>> x.tolist()
    [False, True, True]
    '''
    res = np.sum(data >= cutoff, axis=axis) / data.shape[axis] >= fraction
    if issparse(data):
        res = res.A1
    return res


def unique_cut(x, unique=0.05):
    '''the percentage of distinct values out of the number of total samples.

    Examples
    --------
    >>> unique_cut([0, 0], 0.49)
    True
    >>> unique_cut([0, 0], 0.51)
    False
    >>> unique_cut([0, 1], 1.01)
    False
    '''
    count = len(set(x))
    return count / len(x) >= unique


def freq_ratio(x, ratio=2):
    '''the ratio of the counts of the most common value to the second most common value

    Return True if the ratio is not greater than "ratio".

    Examples
    --------
    >>> freq_ratio([0, 0, 1, 2], 2)
    True
    >>> freq_ratio([0, 0, 1, 1], 1.01)
    True
    >>> freq_ratio([0, 0, 1, 2], 1.99)
    False
    '''
    unique, counts = np.unique(np.array(x), return_counts=True)
    max_1, max_2 = nlargest(2, counts)
    return max_1 / max_2 <= ratio


@Experiment._record_sig
def filter_samples(exp: Experiment, field, values, negate=False, inplace=False):
    '''Shortcut for filtering samples.

    Parameters
    ----------
    field : str
        the column name of the sample metadata tables
    values :
        keep the samples with the values in the given field
    negate : bool, optional
        discard instead of keep the samples if set to `True`
    inplace : bool, optional
        return the filtering on the original :class:`.Experiment` object or a copied one.

    Returns
    -------
    Experiment
        the filtered object

    '''
    values = _to_list(values)
    return filter_by_metadata(exp, field=field, select=values, negate=negate, inplace=inplace)


@ds.with_indent(4)
@Experiment._record_sig
def filter_mean_abundance(exp: Experiment, cutoff=0.01, field=None, **kwargs):
    '''Filter features with a mean at least cutoff of the mean total abundance/sample

    For example, to keep features with mean abundance of 1% use `filter_abundance(cutoff=0.01)`.

    Parameters
    ----------
    cutoff : float, optional
        The minimal mean abundance (in fraction) for a feature in order to keep
        it. Default is 0.01 - keep features with mean abundance >= 1%
        over all samples.
    field : str or `None`, optional
        The column in the sample_metadata. If it is not `None`, the
        data set are divided into groups according to the sample
        metadata column. The features that has mean abundance lower
        than the cutoff in *ALL* sample groups will be filtered away.
        If it is `None`, the mean abundance is computed over the whole
        data set.

    Keyword Arguments
    -----------------
    %(filtering.filter_by_data.parameters.negate)s

    Returns
    -------
    Experiment

    See Also
    --------
    filter_by_data

    '''
    if exp.normalized <= 0:
        logger.warning('Do you forget to normalize your data? It is required before running this function')

    cutoff = exp.normalized * cutoff

    return exp.filter_by_data('mean_abundance', axis=1, field=field, cutoff=cutoff, **kwargs)


@ds.with_indent(4)
@Experiment._record_sig
def filter_abundance(exp: Experiment, cutoff=10, **kwargs):
    '''Filter features with sum abundance across all samples less than the cutoff.

    For example, to keep features with mean abundance of 1% use `filter_abundance(cutoff=0.01)`.

    Parameters
    ----------
    cutoff : float, optional
        The minimal total abundance across all samples.
        Default is 10 - keep features with total abundance >= 10.

    Keyword Arguments
    -----------------
    %(filtering.filter_by_data.parameters.negate)s

    Returns
    -------
    Experiment

    See Also
    --------
    filter_by_data

    '''
    if exp.normalized <= 0:
        logger.warning('Do you forget to normalize your data? It is required before running this function')

    return exp.filter_by_data('sum_abundance', axis=1, field=None, cutoff=cutoff, **kwargs)


@ds.with_indent(4)
@Experiment._record_sig
def filter_prevalence(exp: Experiment, fraction, cutoff=0.01, field=None, **kwargs):
    '''Filter features keeping only ones present in more than certain fraction of all samples.

    This is a convenience function wrapping `filter_by_data`

    Parameters
    ----------
    fraction : float
        Keep features present in more than `fraction` of samples
    cutoff : float, optional
        The min abundance threshold to be called present in a sample
    field : str or `None`, optional
        The column in the sample_metadata. If it is not `None`, the
        data set are divided into groups according to the sample
        metadata column. The features that has prevalence lower
        than the fraction in *ALL* sample groups will be filtered away.
        If it is `None`, the prevalence is computed over the whole
        data set.

    Keyword Arguments
    -----------------
    %(filtering.filter_by_data.parameters.negate)s

    Returns
    -------
    Experiment
        with only features present in at least fraction of samples

    See Also
    --------
    filter_by_data
    filter_mean_abundance
    '''
    if exp.normalized <= 0:
        logger.warning('Do you forget to normalize your data? It is required before running this function')

    return exp.filter_by_data('prevalence', axis=1, field=None, cutoff=cutoff, **kwargs)


@Experiment._record_sig
def filter_ids(exp: Experiment, ids, axis=1, negate=False, inplace=False):
    '''Filter samples or features based on a list IDs.

    .. note:: the order of samples or features is updated as the order given in `ids`.

    Parameters
    ----------
    ids : iterable of str
        the feature/sample ids to filter (index values)
    axis : 0, 1, 's', or 'f', optional
        1 or 'f' (default) to filter features; 0 or 's' to filter samples
    negate : bool, optional
        negate the filtering
    inplace : bool, optional
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    Experiment
        filtered so contains only features/samples present in exp and in ids
    '''
    if axis == 0:
        index = exp.sample_metadata.index
    else:
        index = exp.feature_metadata.index
    try:
        ids_pos = [index.get_loc(i) for i in ids]
    except KeyError as e:
        raise ValueError('Unknown IDs provided: %s' % str(e))
    # use reprlib to shorten the list if it is too long
    logger.debug('Filter by IDs %s on axis %d' % (reprlib.repr(ids), axis))
    if negate:
        ids_pos = np.setdiff1d(np.arange(len(index)), ids_pos, assume_unique=True)
    newexp = exp.reorder(ids_pos, axis=axis, inplace=inplace)
    return newexp
