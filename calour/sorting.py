'''
sorting (:mod:`calour.sorting`)
===============================

.. currentmodule:: calour.sorting

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   reorder
   sort_by_data
   sort_by_metadata
   sort_samples
   sort_abundance
   sort_ids
   cluster_data
   cluster_features
   sort_centroid
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
from scipy import cluster, spatial

from . import Experiment
from .filtering import _filter_ids
from .transforming import log_n, standardize
from .manipulation import chain
from .util import _argsort
from ._doc import ds


logger = getLogger(__name__)


def reorder(exp: Experiment, new_order, axis=0, inplace=False) -> Experiment:
    '''Reorder according to indices in the new order.

    Note that we can also drop samples in new order.

    Parameters
    ----------
    new_order : Iterable of int or boolean mask
        the order of new indices
    axis : 0, 1, 's', or 'f'
        the axis where the reorder occurs. 0 or 's' means reodering samples;
        1 or 'f' means reordering features.
    inplace : bool, optional
        reorder in place.

    Returns
    -------
    Experiment
        experiment with reordered samples
    '''
    if inplace is False:
        exp2 = exp.copy()
    else:
        exp2 = exp
    # make it a np array; otherwise the slicing won't work if the new_order is
    # a list of boolean and data is sparse matrix. For example:
    # from scipy.sparse import csr_matrix
    # a = csr_matrix((3, 4), dtype=np.int8)
    # In [125]: a[[False, False, False], :]
    # Out[125]:
    # <3x4 sparse matrix of type '<class 'numpy.int8'>'

    # In [126]: a[np.array([False, False, False]), :]
    # Out[126]:
    # <0x4 sparse matrix of type '<class 'numpy.int8'>'

    # if new_order is empty, we want to return empty exp2eriment
    # it doesn't work for dense data if we use np.array([]) for the indexing
    if len(new_order) > 0:
        new_order = np.array(new_order)
    if axis == 0:
        exp2.data = exp2.data[new_order, :]
        exp2.sample_metadata = exp2.sample_metadata.iloc[new_order, :]
    else:
        exp2.data = exp2.data[:, new_order]
        if exp2.feature_metadata is not None:
            exp2.feature_metadata = exp2.feature_metadata.iloc[new_order, :]
    return exp2


def sort_centroid(exp: Experiment, transform=log_n, inplace=False, **kwargs) -> Experiment:
    r'''Sort the features based on the center of mass.

    Assuming that samples are already sorted by a field of continuous
    value in sample metadata (eg pH), this function will sort features
    based on their center of mass along this field. Specifically, it
    calculates the center of mass for each feature and sort features
    by their center of mass.

    Parameters
    ----------
    transform : callable, optional
        a callable transform on a 2-d matrix. Input and output of
        transform are :class:`.Experiment`.  The transform function
        modifies :attr:`.Experiment.data` but does not change the
        dimension of :attr:`.Experiment.data`.
    inplace : bool, optional
        False (default) to create a copy
        True to Replace data in exp
    kwargs : dict, optional
        keyword arguments passing to the transformer.

    Returns
    -------
    Experiment
        with features sorted by center of mass

    See Also
    --------
    log_n
    '''
    logger.debug('sorting features by center of mass')
    if transform is None:
        data = exp.data
    else:
        logger.debug('transforming data using %r' % transform)
        data = transform(deepcopy(exp), **kwargs).data
    data = data.T
    center_mass = data.dot(np.arange(0, data.shape[1]))
    s = data.sum(axis=1)
    if isinstance(s, np.matrix):
        s = s.A1
    center_mass = np.divide(center_mass, s)
    sort_pos = np.argsort(center_mass, kind='mergesort')
    exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
    return exp


@ds.get_sectionsf('sorting.cluster_data')
def cluster_data(exp: Experiment, transform=None, axis=1,
                 metric='euclidean', inplace=False, **kwargs) -> Experiment:
    r'''Cluster the samples/features.

    Reorder the features/samples so that ones with similar behavior (pattern
    across samples/features) are close to each other

    Parameters
    ----------
    aixs : 0, 1, 's', or 'f', optional
        'f' or 1 (default) means clustering features; 's' or 0 means clustering samples
    transform : Callable
        a callable that transforms on a 2-d matrix. Its 1st argument
        and return should be both :class:`.Experiment` type. It can
        modify a copy of `Experiment.data` (but should not change its
        dimension.) for clustering.
    metric : str or callable
        the clustering metric to use. It should be able to be passed to
        ``scipy.spatial.distance.pdist``.
    inplace : bool, optional
        False (default) to create a copy.
        True to Replace data in exp.
    kwargs : dict, optional
        keyword arguments passing to the transformer.

    Returns
    -------
    Experiment
        With samples/features clustered (reordered)

    See Also
    --------
    cluster_features

    '''
    logger.debug('clustering data on axis %s' % axis)
    if transform is None:
        data = exp.get_data(sparse=False)
    else:
        logger.debug('transforming data using %r' % transform)
        data = transform(exp, **kwargs, inplace=False).get_data(sparse=False)

    if axis == 1:
        data = data.T
    # cluster
    dist_mat = spatial.distance.pdist(data, metric=metric)
    linkage = cluster.hierarchy.single(dist_mat)
    sort_order = cluster.hierarchy.leaves_list(linkage)

    return exp.reorder(sort_order, axis=axis, inplace=inplace)


def cluster_features(exp: Experiment, cutoff=0, inplace=False) -> Experiment:
    '''Cluster features.

    This function does these things in order:

    1. filter away features that have sum abundance less than cutoff;

    2. cluster features based on transformed data with :func:`.log_n` and :func:`.standardize`.

    3. return the experiment with features clustered.

    .. note:: The transformation is done on a copy of data and used
    for clustering computation. The real :attr:`Experiment.data` is
    not transformed.

    Parameters
    ----------
    cutoff : numeric, optional
        filter away features with sum abundance less than ``cutoff``. Default to 0.

    Returns
    -------
    Experiment

    See Also
    --------
    cluster_data
    log_n
    standardize

    '''
    newexp = exp.filter_sum_abundance(cutoff, inplace=inplace)
    return newexp.cluster_data(transform=chain, steps=[log_n, standardize],
                               standardize__axis=1,
                               axis=1, inplace=True)


@ds.get_sectionsf('sorting.sort_by_metadata')
def sort_by_metadata(exp: Experiment, field, axis=0, inplace=False, reverse=False) -> Experiment:
    '''Sort samples or features based on metadata values in the given field.

    Parameters
    ----------
    field : str
        column name of the sample or feature metadata to sort by.
    axis : 0, 1, 's', or 'f'
        sort by samples (0 or 's') or by features (1 or 'f'), i.e. the ``field`` is a column
        in ``sample_metadata`` (0 or 's') or ``feature_metadata`` (1 or 'f')
    inplace : bool, optional
        False (default) to create a copy
        True to Replace data in exp

    Returns
    -------
    Experiment
    '''
    logger.debug('sorting samples by field %s' % field)
    if axis == 0:
        x = exp.sample_metadata
    elif axis == 1:
        x = exp.feature_metadata

    idx = _argsort(x[field].values, reverse)
    return exp.reorder(idx, axis=axis, inplace=inplace)


@ds.with_indent(4)
def sort_samples(exp: Experiment, field, **kwargs) -> Experiment:
    '''Sort samples by sample metadata.

    A convenience function wrapping on :func:`sort_by_metadata`.

    Parameters
    ----------
    field : str
        The column name in sample metadata to sort the samples by.

    Keyword Arguments
    -----------------
    %(sorting.sort_by_metadata.parameters)s

    Returns
    -------
    Experiment
        with samples sorted according to values in field.

    See Also
    --------
    sort_by_metadata
    '''
    return exp.sort_by_metadata(field=field, axis='s', **kwargs)


@ds.get_sectionsf('sorting.sort_by_data')
def sort_by_data(exp: Experiment, axis=0, subset=None, key='log_mean',
                 inplace=False, reverse=False, **kwargs) -> Experiment:
    '''Sort features based on their values in the data table.

    Sort the 2-d array by sample (axis=0) or feature (axis=0). ``key``
    will be applied to ``subset`` of each feature (axis=0) or sample
    (axis=1) and return a comparative value.

    Parameters
    ----------
    axis : 0, 1, 's', or 'f'
        Apply ``key`` function on row (sort the samples) (0 or 's') or column (sort the features) (1 or 'f')
    subset : {boolean mask, :class:`slice`, int indices}, default=None
        Sorting using only subset of the data. The subsetting occurs on the opposite of
        the specified axis. Default is to use the whole data set.
    key : str or callable
        If it is a callable, it should be a function accepts 1-D array
        of numeric and returns a comparative value (like ``key`` in the
        builtin :func:`sorted`). For example, you can use :func:`numpy.mean` or
        :func:`numpy.media`. Alternatively it accepts the
        following strings for pre-defined functions:

        * 'log_mean': sort by the mean of the log;
        * 'prevalence': sort by the prevalence;

    inplace : bool, default=False
        False to create a copy. True to modify in place.
    reverse : bool, optional
        True to reverse the order of the sort. Similar to :func:`sorted`
    kwargs : dict
        keyword parameters passed to ``key``

    Returns
    -------
    Experiment
    '''
    if subset is None:
        data_subset = exp.data
    else:
        if axis == 0:
            # sort samples, but subset on features
            data_subset = exp.data[:, subset]
        else:
            data_subset = exp.data[subset, :]

    func = {'log_mean': _log_n_1d,
            'prevalence': _prevalence_1d}
    key = func.get(key, key)

    if exp.sparse:
        n = data_subset.shape[axis]
        values = np.zeros(n, dtype=float)
        if axis == 0:
            for row in range(n):
                values[row] = key(data_subset[row, :], **kwargs)
        elif axis == 1:
            for col in range(n):
                values[col] = key(data_subset[:, col], **kwargs)
        sort_pos = np.argsort(values, kind='mergesort')
    else:
        sort_pos = np.argsort(np.apply_along_axis(key, 1 - axis, data_subset, **kwargs), kind='mergesort')

    if reverse:
        sort_pos = sort_pos[::-1]
    exp = exp.reorder(sort_pos, axis=axis, inplace=inplace)

    return exp


def _log_n_1d(x, n=1):
    '''Log transform and then return the mean.

    Examples
    --------
    >>> x = np.array([0, 0, 2, 4])
    >>> _log_n_1d(x)
    0.75
    >>> _log_n_1d(x, 2)
    1.25
    '''
    try:
        x = x.todense().A1
    except AttributeError:
        # make a copy because it changes inplace
        x = np.copy(x)
    x[x < n] = n
    return np.log2(x).mean()


def _prevalence_1d(x, cutoff=0):
    return np.sum(i >= cutoff for i in x) / len(x)


@ds.with_indent(4)
def sort_abundance(exp: Experiment, subgroup=None, **kwargs) -> Experiment:
    '''Sort features based on their abundances in a subset of the samples.

    This is a convenience wrapper for :func:`sort_by_data`.

    Parameters
    ----------
    subgroup : dict, default=None
        ``None`` to sort based on all samples. Subset samples by
        columns (specified by dict keys) in sample metadata matching
        the dict values (a list). sorting is only on samples matching this list.

    Keyword Arguments
    -----------------
    %(sorting.sort_by_data.parameters)s

    Returns
    -------
    Experiment
        with features sorted by abundance

    Examples
    --------
    This selects the samples of week 6 treated either with drugA or drugB
    and use these samples for sorting:
    >>> exp.sort_abundance(subgroup={'treatment': ['drugA', 'drugB'],
    ...                              'week'     : [6]})    # doctest: +SKIP

    See Also
    --------
    sort_by_data
    '''
    if subgroup is None:
        select = None
    else:
        select = np.ones(exp.shape[0], dtype='?')
        for k, v in subgroup.items():
            select_i = np.logical_and(select, exp.sample_metadata[k].isin(v).values)
            select = select & select_i
    return exp.sort_by_data(axis=1, subset=select, **kwargs)


def sort_ids(exp: Experiment, ids, axis=1, inplace=False) -> Experiment:
    '''Sort the features or samples by the given ids.

    If ``ids`` does not cover the all the features (samples), the rest
    will be unsorted and appended.  If ``ids`` has duplicates, the
    resulting Experiment will also have duplicates. If ``ids`` is
    unordered data type (eg set type), the resulting Experiment is
    also unordered.

    Parameters
    ----------
    ids : sequence of str
        The ids to put first in the resulting Experiment. If ``ids``
        is unordered data type (eg set type), the resulting Experiment
        is also unordered for those ids.
    axis : {0, 1, 's', 'f'}
        sort by samples (0 or 's') or by features (1 or 'f'), i.e. the ``field`` is a column name
        in ``sample_metadata`` (0 or 's') or ``feature_metadata`` (1 or 'f')
    inplace : bool, default=False
        False to create a copy of the experiment; True to filter inplace

    Returns
    -------
    Experiment
        with features/samples first according to ``ids`` and then the rest.

    '''
    ids_pos = _filter_ids(exp, ids, axis)
    rest_pos = np.setdiff1d(np.arange(exp.shape[axis]), ids_pos, assume_unique=True)
    newexp = exp.reorder(np.concatenate([ids_pos, rest_pos]), axis=axis, inplace=inplace)
    return newexp
