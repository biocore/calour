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
from . import transforming
from .transforming import log_n


logger = getLogger(__name__)


@Experiment._record_sig
def sort_center_mass(exp, transform=log_n, inplace=False, **kwargs):
    '''Sort the features based on the center of mass

    Assumes exp samples are sorted by some continuous field, and sort the features based on their
    center of mass along this field order:
    For each feature calculate the center of mass (i.e. for each feature go over all samples i and calculate sum(data(i) * i / sum(data(i)) ).
    Features are then sorted according to this center of mass

    Parameters
    ----------
    transform : callable (optional)
        a callable transform on a 2-d matrix. Input and output of transform are ``Experiment``.
        The transform function can modify ``Experiment.data`` (it is a copy).
        It should not change the dimension of ``data`` in ``Experiment``.
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp
    kwargs : dict
        additional keyword parameters passed to ``transform``.

    Returns
    -------
    exp : Experiment
        features sorted by center of mass
    '''
    logger.debug('sorting features by center of mass')
    if transform is None:
        data = exp.data
    else:
        logger.debug('tansforming data using %r' % transform)
        newexp = deepcopy(exp)
        data = transform(newexp, **kwargs).data
    data = data.T
    coordinates = np.arange(data.shape[1]) - ((data.shape[1] - 1)/2)
    center_mass = data.dot(coordinates)
    center_mass = np.divide(center_mass, data.sum(axis=1))

    sort_pos = np.argsort(center_mass, kind='mergesort')
    exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
    return exp


@Experiment._record_sig
def cluster_data(exp, transform=None, axis=0, metric='euclidean', inplace=False, **kwargs):
    '''Cluster the samples/features.

    Reorder the features/samples so that ones with similar behavior (pattern
    across samples/features) are close to each other

    Parameters
    ----------
    aixs : 0 or 1 (optional)
        0 (default) means clustering features; 1 means clustering samples
    transform : Callable
        a callable transform on a 2-d matrix. Input and output of transform are ``Experiment``.
        The transform function can modify ``Experiment.data`` (it is a copy).
        It should not change the dimension of ``data`` in ``Experiment``.
    metric : str or callable
        the clustering metric to use. It should be able to be passed to
        ``scipy.spatial.distance.pdist``.
    inplace : bool (optional)
        False (default) to create a copy.
        True to Replace data in exp.
    kwargs : dict
        additional keyword parameters passed to ``transform``.

    Returns
    -------
    exp : Experiment
        With samples/features clustered (reordered)

    '''
    logger.debug('clustering data on axis %s' % axis)
    if transform is None:
        data = exp.get_data(sparse=False)
    else:
        logger.debug('tansforming data using %r' % transform)
        newexp = deepcopy(exp)
        data = transform(newexp, **kwargs).get_data(sparse=False)

    if axis == 0:
        data = data.T
    # cluster
    dist_mat = spatial.distance.pdist(data, metric=metric)
    linkage = cluster.hierarchy.single(dist_mat)
    sort_order = cluster.hierarchy.leaves_list(linkage)

    return exp.reorder(sort_order, axis=1 - axis, inplace=inplace)


@Experiment._record_sig
def sort_by_metadata(exp, field, axis=0, inplace=False):
    '''Sort samples or features based on metadata values in the field.

    Parameters
    ----------
    field : str
        Name of the field to sort by
    axis : 0 or 1
        sort by samples (0) or by features (1), i.e. the ``field`` is a column
        in ``sample_metadata`` (0) or ``feature_metadata`` (1)
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp

    Returns
    -------
    ``Experiment``
    '''
    logger.info('sorting samples by field %s' % field)
    if axis == 0:
        x = exp.sample_metadata
    elif axis == 1:
        x = exp.feature_metadata
    else:
        raise ValueError('unknown axis %s' % axis)
    idx = np.argsort(x[field].values, kind='mergesort')
    return exp.reorder(idx, axis=axis, inplace=inplace)


@Experiment._record_sig
def sort_by_data(exp, axis=0, subset=None, key='log_mean', inplace=False, **kwargs):
    '''Sort features based on their mean frequency.

    Sort the 2-d array by sample (axis=0) or feature (axis=0). ``key``
    will be applied to ``subset`` of each feature (axis=0) or sample
    (axis=1) and return a comparative value.

    Parameters
    ----------
    axis : 0 or 1
        Apply ``key`` function on row (sort the samples) (0) or column (sort the features) (1)
    subset : ``None`` or iterable of int (optional)
        Sorting using only subset of the data.
    key : str or callable
        If it is a callable, it should be a function accepts 1-D array of numeric and
        returns a comparative value (like ``key`` in builtin ``sorted`` function).
        Alternatively it accepts the following str values:
        "log_mean": sort by log of the mean
        "prevalence": sort by the prevalence
        "mean": sort by the mean
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp
    kwargs : dict
        key word parameters passed to ``key``

    Returns
    -------
    ``Experiment``
        With features sorted by mean frequency

    '''
    if subset is None:
        data_subset = exp.data
    else:
        if axis == 0:
            data_subset = exp.data[:, subset]
        else:
            data_subset = exp.data[subset, :]
    func = {'log_mean': _log_mean,
            'prevalence': _prevalence,
            'mean': np.mean}
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


def sort_samples(exp, field, **kwargs):
    '''Sort samples by field
    A convenience function for sort_by_metadata

    Parameters
    ----------
    field : str
        The field to sort the samples by

    Returns
    -------
    ``Experiment`` with samples sorted according to values in field
    '''
    newexp = exp.sort_by_metadata(field=field, **kwargs)
    return newexp


def sort_abundance(exp, field=None, value=None, inplace=False, **kwargs):
    '''Sort features based on their abundance in a subset of the samples.
    This is a convenience wrapper for sort_by_data()

    Parameters
    ----------
    field : str or None (default)
        None (default) to sort on all samples, str to sort only on samples matching the field/value combination
    value : str or list of str or None (default)
        if field is not None, value is the value/list of values so sorting is only on samples matching this list
    inplace : bool (optional)
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    ``Experiment``
        with features sorted by abundance
    '''
    if field is None:
        subset = None
    else:
        if not isinstance(value, (list, tuple)):
            value = [value]
        subset = np.where(exp.sample_metadata[field].isin(value).values)[0]

    newexp = exp.sort_by_data(axis=1, subset=subset, inplace=inplace, **kwargs)
    return newexp


def cluster_features(exp, min_abundance=10, inplace=False, **kwargs):
    '''Cluster features following filtering of minimal abundance on features and
       log transform and scaling on the data (mean 0 std 1)
    '''
    if min_abundance > 0:
        newexp = exp.filter_min_abundance(min_abundance, inplace=inplace)
    else:
        newexp = exp
    newexp = newexp.cluster_data(transform=transforming.transform, axis=0, steps=[transforming.log_n, transforming.scale], scale__axis=0, inplace=inplace, **kwargs)
    return newexp
