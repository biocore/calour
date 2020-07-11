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
   filter_features
   filter_ids
   filter_prevalence
   filter_sum_abundance
   filter_mean_abundance
   filter_sample_group
   downsample
   is_abundant
   is_prevalent
   freq_ratio
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
from collections.abc import Callable
import reprlib

import numpy as np
from scipy.sparse import issparse

from . import Experiment
from ._doc import ds
from .util import _to_list


logger = getLogger(__name__)


def downsample(exp: Experiment, field, axis=0, keep=None,
               inplace=False, random_seed=None) -> Experiment:
    '''Downsample the data set.

    This down samples all the samples/features to have the same number of
    samples/features for each categorical value of the field in
    ``sample_metadata`` or ``feature_metadata``.

    Parameters
    ----------
    field : str
        The name of the column in samples metadata table. This column
        should has categorical values
    axis : 0, 1, 's', or 'f', optional
        0 or 's' (default) to filter samples; 1 or 'f' to filter features
    keep : int, default=None
        Downsample to keep samples/features per group. If a group has
        samples/features smaller than ``keep``, the whole group is dropped.
        Default to downsample to minimal group size.
    inplace : bool, optional
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    See Also
    --------
    filter_sample_group
    '''
    logger.debug('downsample on field %s' % field)
    if axis == 0:
        x = exp.sample_metadata
        axis_name = 'sample'
    elif axis == 1:
        x = exp.feature_metadata
        axis_name = 'feature'

    if field not in x:
        raise ValueError('Field %s not in %s_metadata (existing fields are: %s)' % (field, axis_name, x.columns))

    # convert to string type because nan values, if they exist in the column,
    # will fail `np.unique`
    values = x[field].astype(str).values
    keep = _balanced_subsample(values, keep, random_seed)
    return exp.reorder(keep, axis=axis, inplace=inplace)


def _balanced_subsample(x, n=None, random_seed=None):
    '''subsample the array to have equal number count for each unique values.

    Parameters
    ----------
    x : array
    n : int. count
    random_seed : int, np.radnom.Generator instance or None, optional, default=None

    Returns
    -------
    array of bool
    '''
    rng = np.random.default_rng(random_seed)
    keep = np.zeros(x.shape[0], dtype='?')
    unique, counts = np.unique(x, return_counts=True)
    if n is None:
        n = counts.min()
    for value in unique:
        i_indice = np.where(x == value)[0]
        if i_indice.shape[0] >= n:
            idx = rng.choice(i_indice, n, replace=False)
            keep[idx] = True
    return keep


def filter_sample_group(exp: Experiment, field, min_samples=5, inplace=False) -> Experiment:
    '''Drop sample groups that have too few samples.

    This is useful to get rid of groups with too few samples for
    supervised classification training or other stat analyses because
    of insufficient statistical power. It also drops the samples that
    don't have unspecified group.

    Examples
    --------

    Parameters
    ----------
    field : str
        The name of the column in samples metadata table. This column
        should has categorical values
    min_samples : int, optional
        Filter away the group of samples if its sample size is
        less than min_samples.
    inplace : bool, optional
        False (default) to create a copy of the experiment, True to filter inplace

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


def filter_by_metadata(exp: Experiment, field, select, axis=0,
                       negate=False, inplace=False) -> Experiment:
    '''Filter samples or features by metadata.

    Parameters
    ----------
    field : str
        the column name of the sample or feature metadata tables
    select : None, Callable, or list/set/tuple-like
        select what to keep based on the value in the specified field.

        * callable: the callable accepts a 1D array and return a
          boolean array of the same length;
        * list/set/tuple-like object: keep the samples with the values in
          the `field` column included in the `select`;
        * ``None``: filter out the NA.

    axis : 0, 1, 's', or 'f', optional
        the field is on samples (0 or 's') or features (1 or 'f') metadata
    negate : bool, optional
        discard instead of keep the select if set to `True`
    inplace : bool, optional
        do the filtering on the original :class:`.Experiment` object or a copied one.

    Returns
    -------
    Experiment
        the filtered experiment
    '''
    if axis == 0:
        metadata = exp.sample_metadata
        axis_name = 'sample'
    elif axis == 1:
        metadata = exp.feature_metadata
        axis_name = 'feature'

    if field not in metadata:
        raise ValueError('Field %s not in %s_metadata. (fields are: %s)' % (field, axis_name, metadata.columns))

    if isinstance(select, Callable):
        select = select(metadata[field])
    elif select is None:
        select = metadata[field].notnull()
    else:
        select = metadata[field].isin(select).values

    if negate is True:
        select = ~ select
    return exp.reorder(select, axis=axis, inplace=inplace)


@ds.get_sectionsf('filtering.filter_by_data')
def filter_by_data(exp: Experiment, predicate, axis=1, field=None,
                   negate=False, inplace=False, **kwargs) -> Experiment:
    '''Filter samples or features by the data matrix.

    Parameters
    ----------
    predicate : str or callable
        The callable accepts a list of numeric and return a bool. Alternatively
        it also accepts the following strings to filter along the specified axis:

        * 'abundance': calls :func:`is_abundant`, filter by abundance;
        * 'prevalence': calls :func:`is_prevalent`, filter by prevalence;
        * 'freq_ratio': calls :func:`freq_ratio`, filter if there is a dominant unique value;
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
        keyword argument passing to predicate function.

    See Also
    --------
    filter_mean_abundance
    filter_sum_abundance
    filter_prevalence

    '''
    if exp.normalized <= 0:
        logger.warning('Do you forget to normalize your data? It is required before running this function')
    logger.debug('filter_by_data using function %r' % predicate)

    if axis == 0:
        metadata = exp.feature_metadata
        # transpose it so all the following operations are performed on column
        data = exp.data.T
    elif axis == 1:
        metadata = exp.sample_metadata
        data = exp.data

    if field is None:
        groups = [0]
        indices = np.zeros(data.shape[0])
    else:
        values = metadata[field].values.astype('U')
        groups, indices = np.unique(values, return_inverse=True)

    # functions that can be applied to 2d matrix
    func = {'abundance': is_abundant,
            'prevalence': is_prevalent,
            'freq_ratio': freq_ratio}
    pred = func.get(predicate, predicate)

    n = data.shape[1]
    select = np.zeros(n, dtype='?')

    for i, _ in enumerate(groups):
        select_i = pred(data[indices == i], axis=0, **kwargs)
        select = select | select_i

    if negate is True:
        select = ~ select

    logger.info('After filtering, %s remain.' % np.sum(select))
    return exp.reorder(select, axis=axis, inplace=inplace)


def is_abundant(data, axis, cutoff=0.01, strict=False, mean_or_sum='mean'):
    '''Check if the mean or sum abundance larger than cutoff.

    Can be used to keep features with means at least "cutoff" in all
    samples

    Parameters
    ----------
    data : 1D or 2D numpy.ndarray or scipy.sparse.csr_matrix
    axis : int
        0 to average each column, 1 to average each row. passed to :func:`numpy.mean`
    cutoff : float
        the abundance threshold
    strict : bool, default=False
        False to use abundance >= cutoff; True to use abundance > cutoff
    mean_or_sum : {'mean', 'sum'}
        what abundance to compute

    Returns
    -------
    bool or 1D np.ndarray of bool
        If the input data is 1D, return True if the mean/sum abundance >= cutoff; otherwise, False.
        If the input data is 2D, return array with boolean value for each row or column.

    Examples
    --------
    >>> is_abundant(np.array([[0, 0, 1], [1, 1, 1]]), axis=1, cutoff=0.51).tolist()
    [False, True]
    >>> is_abundant(np.array([0, 0, 1, 1]), axis=0, cutoff=0.5)
    True
    >>> is_abundant(np.array([0, 0, 1, 1]), axis=0, cutoff=0.5, strict=True)
    False
    >>> is_abundant(np.array([[0, 1, 1]]), axis=1, cutoff=2, mean_or_sum='sum').tolist()
    [True]
    >>> is_abundant(np.array([0, 1, 1]), axis=0, cutoff=2, strict=True, mean_or_sum='sum')
    False
    >>> is_abundant(np.array([[0, 1, 1]]), axis=1, cutoff=2.01, mean_or_sum='sum').tolist()
    [False]
    '''
    if mean_or_sum == 'mean':
        m = data.mean(axis=axis)
    elif mean_or_sum == 'sum':
        m = data.sum(axis=axis)
    if strict is True:
        res = m > cutoff
    else:
        res = m >= cutoff
    if issparse(data):
        res = res.A1
    return res


def is_prevalent(data, axis, cutoff=1, fraction=0.1):
    '''Check the prevalent of values above the cutoff.

    Return `True` if its values are >= `cutoff` in at least `frac`
    fraction of samples.

    Parameters
    ----------
    data : 1D or 2D numpy.ndarray or scipy.sparse.csr_matrix
    axis : int
        compute prevalence of each column (0) or row (1).
    cutoff : float
        the min threshold of abundance
    fraction : float
        [0, 1). the min threshold of presence (in fraction)

    Returns
    -------
    bool or 1D np.ndarray of bool
        If the input data is 1D, return True if the prevalence >= cutoff; otherwise, False.
        If the input data is 2D, return array with boolean value for each row or column.

    Examples
    --------
    >>> a = np.array([[0, 1, 2],
    ...               [0, 2, 2]])
    >>> x = is_prevalent(a, 0, 2, 0.51)
    >>> x.tolist()
    [False, False, True]
    >>> x = is_prevalent(a, 0, 2, 0.5)
    >>> x.tolist()
    [False, True, True]
    >>> x = is_prevalent(a[0], 0, 2, 0.5)
    >>> x.tolist()
    False
    >>> x = is_prevalent(a, 1, 2, 0.66)
    >>> x.tolist()
    [False, True]
    '''
    res = np.sum(data >= cutoff, axis=axis) / data.shape[axis] >= fraction
    if issparse(data):
        res = res.A1
    return res


def freq_ratio(data, axis, ratio=19):
    '''Check if frequency ratios is not too big.

    Frequency ratio is defined as the frequency of the most prevalent
    value over the second most frequent value for a 1D array. This
    function compute the frequency ratios along the given `axis` and
    compare them against the `ratio` (True if it is smaller than
    `ratio`).

    This is modeled after R package `caret`.

    Parameters
    ----------
    data : 2D numpy.ndarray or scipy.sparse.csr_matrix
    axis : int
        compute prevalence of each column (0) or row (1).
    ratio : float

    Returns
    -------
    1D np.ndarray of bools
        It is the same size of input array rows or columns depending on axis.

    Examples
    --------
    >>> data = np.array([[0, 0, 1, 2],
    ...                  [0, 0, 1, 1]])
    >>> x = freq_ratio(data, 1, 1.01)
    >>> x.tolist()
    [False, True]
    >>> import scipy
    >>> data = scipy.sparse.csr_matrix(data)
    >>> x = freq_ratio(data, 1, 1.01)
    >>> x.tolist()
    [False, True]
    >>> x = freq_ratio(data, 0, 1.01)
    >>> x.tolist()
    [False, False, False, True]

    '''
    if issparse(data):
        res = np.ones(data.shape[1-axis], dtype='?')
        if axis == 0:
            data = data.T
        for i, x in enumerate(data):
            x = x.todense().A1
            res[i] = freq_ratio_1d(x, ratio)
    else:
        res = np.apply_along_axis(freq_ratio_1d, axis, data, ratio)
    return res


def freq_ratio_1d(x, ratio):
    '''Check the ratio of the counts of the most common value to the second most common value.

    Return True if the ratio is not greater than "ratio".

    Parameters
    ----------
    x : 1D array
    ratio : float

    Return
    ------
    bool

    Examples
    --------
    >>> freq_ratio_1d([0, 0, 1, 2], 2)
    True
    >>> freq_ratio_1d([0, 0, 1, 1], 1.01)
    True
    >>> freq_ratio_1d([0, 0, 1, 2], 1.99)
    False

    See Also
    --------
    freq_ratio
    '''
    unique, counts = np.unique(np.array(x), return_counts=True)
    if len(unique) == 1:
        # if there is only one unique value in the array
        return False
    max_1, max_2 = nlargest(2, counts)
    return max_1 / max_2 <= ratio


def filter_samples(exp: Experiment, field, values, negate=False, inplace=False) -> Experiment:
    '''A convenience function for filtering samples.

    Parameters
    ----------
    field : str
        the column name of the sample metadata.
    values : {Iterable, None}
        keep the samples with the values in the specified column.
        `None` will remove NaN samples in the specified column.
    negate : bool, optional
        discard instead of keep the samples if set to `True`
    inplace : bool, optional
        change the filtering on the original :class:`.Experiment` object or not.

    Returns
    -------
    Experiment
        the filtered object

    '''
    # if it is None - pass to filter_by_metadata directly so that the NaN samples will be removed.
    if values is not None:
        values = _to_list(values)

    return filter_by_metadata(exp, field=field, select=values, negate=negate, inplace=inplace)


def filter_features(exp: Experiment, field, values, negate=False, inplace=False):
    '''A convenience function for filtering features.

    Parameters
    ----------
    field : str
        the column name of the feature metadata.
    values : {Iterable, None}
        keep the samples with the values in the specified column.
        `None` will remove NaN samples in the specified column.
    negate : bool, optional
        discard instead of keep the samples if set to `True`
    inplace : bool, optional
        change the filtering on the original :class:`.Experiment` object or not.

    Returns
    -------
    Experiment
        the filtered object

    See Also
    --------
    filter_samples
    filter_by_metadata
    '''
    if values is not None:
        values = _to_list(values)

    return filter_by_metadata(exp, field=field, select=values, axis=1, negate=negate, inplace=inplace)


def filter_mean_abundance(exp: Experiment, frac=0.01, field=None,
                          inplace=False, strict=False) -> Experiment:
    '''Filter features with minimum mean abundance.

    For example, to keep features with mean abundance of 1% use `filter_mean_abundance(frac=0.01)`.

    Parameters
    ----------
    frac : float, optional
        The minimum mean abundance (*in fraction*) for a feature in order to keep
        it. Default is 0.01 - keep features with mean abundance >= 1%
        over all samples.
    field : str or `None`, optional
        The column in the sample_metadata. If it is not `None`, the
        data set are divided into groups according to the sample
        metadata column. The features that has mean abundance lower
        than the cutoff in *ALL* sample groups will be filtered away.
        If it is `None`, the mean abundance is computed over the whole
        data set.
    strict : bool, default=False
        False to use abundance >= cutoff; True to use abundance > cutoff

    Returns
    -------
    Experiment

    See Also
    --------
    filter_by_data
    filter_sum_abundance
    '''
    cutoff = exp.normalized * frac

    return exp.filter_by_data('abundance', axis=1, field=field, inplace=inplace,
                              cutoff=cutoff, mean_or_sum='mean', strict=strict)


def filter_sum_abundance(exp: Experiment, cutoff=10, field=None,
                         inplace=False, strict=False) -> Experiment:
    '''Filter features with minimum sum abundance.

    Parameters
    ----------
    cutoff : float, optional
        The minimum total abundance across all samples.
        Default is 10 - keep features with total abundance >= 10.
    field : str or `None`, optional
        The column in the sample_metadata. If it is not `None`, the
        data set are divided into groups according to the sample
        metadata column. The features that has mean abundance lower
        than the cutoff in *ALL* sample groups will be filtered away.
        If it is `None`, the mean abundance is computed over the whole
        data set.
    strict : bool, default=False
        False to use abundance >= cutoff; True to use abundance > cutoff

    Returns
    -------
    Experiment

    See Also
    --------
    filter_by_data
    filter_mean_abundance
    '''
    return exp.filter_by_data('abundance', axis=1, field=field, inplace=inplace,
                              cutoff=cutoff, mean_or_sum='sum', strict=strict)


def filter_prevalence(exp: Experiment, fraction, cutoff=1, field=None,
                      inplace=False) -> Experiment:
    '''Filter features that are present in more than certain fraction of all samples.

    Parameters
    ----------
    fraction : float
        Keep features present in more than `fraction` of samples
    cutoff : float, optional
        The abundance threshold to be called present in a sample
    field : str or `None`, optional
        The column in the sample_metadata. If it is not `None`, the
        data set are divided into groups according to the sample
        metadata column. The features that has is_prevalent lower
        than the fraction in *ALL* sample groups will be filtered away.
        If it is `None`, the is_prevalent is computed over the whole
        data set.

    Returns
    -------
    Experiment
        with only features present in at least fraction of samples

    See Also
    --------
    filter_by_data
    filter_mean_abundance
    '''
    return exp.filter_by_data('prevalence', axis=1, field=None, inplace=inplace,
                              cutoff=cutoff, fraction=fraction)


def _filter_ids(exp: Experiment, ids, axis):
    if axis == 0:
        index = exp.sample_metadata.index
    else:
        index = exp.feature_metadata.index
    fids = set(ids) & set(index.values)
    if len(fids) < len(ids):
        logger.warning('%d ids were not in the experiment and were dropped.' % (len(ids) - len(fids)))

    ids_pos = [index.get_loc(i) for i in ids if i in fids]

    # use reprlib to shorten the list if it is too long
    logger.debug('Filter by IDs %s on axis %d' % (reprlib.repr(fids), axis))

    return ids_pos


def filter_ids(exp: Experiment, ids, axis=1, negate=False, inplace=False) -> Experiment:
    '''Filter samples or features based on a list IDs.

    If ``ids`` has duplicates, the resulting Experiment will also have
    duplicates. The order of samples or features is updated as the
    order given `ids`, unless ``ids`` is unordered data type (eg set
    type), in which case the resulting Experiment is also unordered.

    Parameters
    ----------
    ids : sequence of str
        The ids to put first in the resulting Experiment.
    axis : {0, 1, 's', 'f'}
        sort by samples (0 or 's') or by features (1 or 'f'), i.e. the ``field`` is a column name
        in ``sample_metadata`` (0 or 's') or ``feature_metadata`` (1 or 'f')
    inplace : bool, default=False
        False to create a copy of the experiment; True to filter inplace
    negate : bool, optional
        negate the filtering
    inplace : bool, optional
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    Experiment
        contains only features/samples present in original experiment and in ids

    '''
    ids_pos = _filter_ids(exp, ids, axis)
    if negate:
        ids_pos = np.setdiff1d(np.arange(exp.shape[axis]), ids_pos, assume_unique=True)

    return exp.reorder(ids_pos, axis=axis, inplace=inplace)
