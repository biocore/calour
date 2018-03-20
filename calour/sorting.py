'''
sorting (:mod:`calour.sorting`)
===============================

.. currentmodule:: calour.sorting

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

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
import reprlib

import numpy as np
from scipy import cluster, spatial

from . import Experiment
from .transforming import log_n, transform, scale
from .util import _argsort
from ._doc import ds


logger = getLogger(__name__)


@ds.with_indent(4)
@Experiment._record_sig
def sort_centroid(exp: Experiment, transform=log_n, inplace=False, **kwargs):
    '''Sort the features based on the center of mass

    Assumes exp samples are sorted by some continuous field, and sort
    the features based on their center of mass along this field order:
    For each feature calculate the center of mass (i.e. for each
    feature go over all samples i and calculate sum(data(i) * i /
    sum(data(i)) ).  Features are then sorted according to this center
    of mass

    Parameters
    ----------
    transform : callable, optional
        a callable transform on a 2-d matrix. Input and output of transform are :class:`.Experiment`.
        The transform function can modify ``Experiment.data`` (it is a copy).
        It should not change the dimension of :attr:`.Experiment.data`.
    inplace : bool, optional
        False (default) to create a copy
        True to Replace data in exp
    kwargs : dict
        additional keyword parameters passed to ``transform``.

    Keyword Arguments
    -----------------
    %(transforming.transform.parameters)s

    Returns
    -------
    Experiment
        features sorted by center of mass

    See Also
    --------
    transform
    '''
    logger.debug('sorting features by center of mass')
    if transform is None:
        data = exp.data
    else:
        logger.debug('transforming data using %r' % transform)
        newexp = deepcopy(exp)
        data = transform(newexp, **kwargs).data
    data = data.T
    coordinates = np.arange(data.shape[1]) - ((data.shape[1] - 1)/2)
    center_mass = data.dot(coordinates)
    center_mass = np.divide(center_mass, data.sum(axis=1))

    sort_pos = np.argsort(center_mass, kind='mergesort')
    exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
    return exp


@ds.get_sectionsf('sorting.cluster_data')
@ds.with_indent(4)
@Experiment._record_sig
def cluster_data(exp: Experiment, transform=None, axis=1, metric='euclidean', inplace=False, **kwargs):
    '''Cluster the samples/features.

    Reorder the features/samples so that ones with similar behavior (pattern
    across samples/features) are close to each other

    Parameters
    ----------
    aixs : 0, 1, 's', or 'f', optional
        'f' or 1 (default) means clustering features; 's' or 0 means clustering samples
    transform : Callable
        a callable transform on a 2-d matrix. Input and output of transform are :class:`.Experiment`.
        The transform function can modify ``Experiment.data`` (it is a copy).
        It should not change the dimension of :attr:`.Experiment.data`.
    metric : str or callable
        the clustering metric to use. It should be able to be passed to
        ``scipy.spatial.distance.pdist``.
    inplace : bool, optional
        False (default) to create a copy.
        True to Replace data in exp.

    Keyword Arguments
    -----------------
    %(transforming.transform.parameters)s

    Returns
    -------
    Experiment
        With samples/features clustered (reordered)

    See Also
    --------
    transform
    '''
    logger.debug('clustering data on axis %s' % axis)
    if transform is None:
        data = exp.get_data(sparse=False)
    else:
        logger.debug('tansforming data using %r' % transform)
        newexp = deepcopy(exp)
        data = transform(newexp, **kwargs).get_data(sparse=False)

    if axis == 1:
        data = data.T
    # cluster
    dist_mat = spatial.distance.pdist(data, metric=metric)
    linkage = cluster.hierarchy.single(dist_mat)
    sort_order = cluster.hierarchy.leaves_list(linkage)

    return exp.reorder(sort_order, axis=axis, inplace=inplace)


@ds.with_indent(4)
@Experiment._record_sig
def cluster_features(exp: Experiment, min_abundance=0, inplace=False, **kwargs):
    '''Cluster features.

    Cluster is done after filtering of minimal abundance, log
    transforming, and scaling on features.

    Parameters
    ----------
    min_abundance : Number, optional
        filter away features less than ``min_abundance``. Default to 0.

    Keyword Arguments
    -----------------
    %(sorting.cluster_data.parameters)s

    Returns
    -------
    Experiment
        object with features filtered, log transformed and scaled.

    See Also
    --------
    cluster_data
    transform
    log_n
    scale

    '''
    newexp = exp.filter_abundance(min_abundance, inplace=inplace)
    return newexp.cluster_data(
        transform=transform,
        axis=1,
        steps=[log_n, scale],
        scale__axis=1,
        inplace=inplace,
        **kwargs)


@ds.get_sectionsf('sorting.sort_by_metadata')
@Experiment._record_sig
def sort_by_metadata(exp: Experiment, field, axis=0, inplace=False):
    '''Sort samples or features based on metadata values in the field.

    Parameters
    ----------
    field : str
        Name of the field to sort by
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
    else:
        raise ValueError('unknown axis %s' % axis)
    idx = _argsort(x[field].values)
    return exp.reorder(idx, axis=axis, inplace=inplace)


@ds.get_sectionsf('sorting.sort_by_data')
@Experiment._record_sig
def sort_by_data(exp: Experiment, axis=0, subset=None, key='log_mean', inplace=False, reverse=False, **kwargs):
    '''Sort features based on their mean frequency.

    Sort the 2-d array by sample (axis=0) or feature (axis=0). ``key``
    will be applied to ``subset`` of each feature (axis=0) or sample
    (axis=1) and return a comparative value.

    Parameters
    ----------
    axis : 0, 1, 's', or 'f'
        Apply ``key`` function on row (sort the samples) (0 or 's') or column (sort the features) (1 or 'f')
    subset : None, boolean mask, :class:`slice`, or int indices, optional
        Sorting using only subset of the data. The subsetting occurs on the opposite of
        the specified axis. Default is to use the whole data set.
    key : str or callable
        If it is a callable, it should be a function accepts 1-D array
        of numeric and returns a comparative value (like ``key`` in
        builtin :func:`sorted`. For example, you can use :func:`numpy.mean` or
        :func:`numpy.media`. Alternatively it accepts the
        following strings:

        * 'log_mean': sort by the mean of the log;

        * 'prevalence': sort by the prevalence;
    inplace : bool, optional
        False (default) to create a copy. True to modify in place.
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

    func = {'log_mean': _log_mean,
            'prevalence': _prevalence}
    if isinstance(key, str):
        key = func[key]
    if exp.sparse:
        n = data_subset.shape[axis]
        values = np.zeros(n, dtype=float)
        if axis == 0:
            for row in range(n):
                values[row] = key(data_subset[row, :], **kwargs)
        elif axis == 1:
            for col in range(n):
                values[col] = key(data_subset[:, col], **kwargs)
        sort_pos = np.argsort(values)
    else:
        sort_pos = np.argsort(np.apply_along_axis(key, 1 - axis, data_subset, **kwargs))

    if reverse:
        sort_pos = sort_pos[::-1]
    exp = exp.reorder(sort_pos, axis=axis, inplace=inplace)

    return exp


def _log_mean(x, logit=1):
    '''Log transform and then return the mean.

    It caps the small value to the `logit` on the lower end before it does
    log2 transform.

    Examples
    --------
    >>> x = np.array([0, 0, 2, 4])
    >>> _log_mean(x)
    0.75
    >>> _log_mean(x, 2)
    1.25
    >>> _log_mean(x, None)  # don't log transform
    1.5
    '''
    if logit is None:
        return np.mean(x)
    else:
        try:
            x = x.toarray()
        except AttributeError:
            # make a copy because it's changed inplace
            x = x.copy()
        x[x < logit] = logit
        return np.log2(x).mean()


def _prevalence(x, cutoff=0):
    return np.sum(i >= cutoff for i in x) / len(x)


@ds.with_indent(4)
@Experiment._record_sig
def sort_samples(exp: Experiment, field, **kwargs):
    '''Sort samples by field
    A convenience function for sort_by_metadata

    Parameters
    ----------
    field : str
        The field to sort the samples by

    Keyword Arguments
    -----------------
    %(sorting.sort_by_metadata.parameters)s

    Returns
    -------
    Experiment with samples sorted according to values in field

    See Also
    --------
    sort_by_metadata
    '''
    newexp = exp.sort_by_metadata(field=field, **kwargs)
    return newexp


@ds.with_indent(4)
@Experiment._record_sig
def sort_abundance(exp: Experiment, subgroup=None, **kwargs):
    '''Sort features based on their abundance in a subset of the samples.

    This is a convenience wrapper for :func:`sort_by_data`.

    Parameters
    ----------
    subgroup : dict or None (default)
        None (default) to sort on all samples. Subset samples by
        columns (specified by dict keys) in sample metadata matching
        the dict values (a list). sorting is only on samples matching this list

    Keyword Arguments
    -----------------
    %(sorting.sort_by_data.parameters)s

    Returns
    -------
    Experiment
        with features sorted by abundance

    See Also
    --------
    sort_by_data
    '''
    if subgroup is None:
        select = None
    else:
        select = np.ones(exp.shape[0])
        for k, v in subgroup.items():
            select = np.logical_and(select, exp.sample_metadata[k].isin(v).values)
    return exp.sort_by_data(axis=1, subset=select, **kwargs)


@Experiment._record_sig
def sort_ids(exp: Experiment, ids, axis=1, inplace=False):
    '''Sort the features or samples by the given ids.

    If ids are not cover the all the features (samples), the rest will be unsorted and appended.

    Parameters
    ----------
    ids : list of str
        The ids to put first in the new experiment
    axis : 0, 1, 's', or 'f'
        sort by samples (0 or 's') or by features (1 or 'f'), i.e. the ``field`` is a column
        in ``sample_metadata`` (0 or 's') or ``feature_metadata`` (1 or 'f')
    inplace : bool, optional
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    Experiment
        with features/samples first according to the ids list and then the rest
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
    logger.debug('Sort by IDs %s on axis %d' % (reprlib.repr(ids), axis))
    rest_pos = np.setdiff1d(np.arange(len(index)), ids_pos, assume_unique=True)
    newexp = exp.reorder(np.concatenate([ids_pos, rest_pos]), axis=axis, inplace=inplace)
    return newexp
