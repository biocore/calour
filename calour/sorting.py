# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import copy

import numpy as np
import scipy.sparse
from scipy import cluster, spatial

from . import Experiment
from .util import _get_taxonomy_string
from .filtering import _filter_by_data
from .transforming import _log_min_transform


logger = getLogger(__name__)


@Experiment._record_sig
def sort_taxonomy(exp, inplace=False):
    '''Sort the features based on the taxonomy

    Sort features based on the taxonomy (alphabetical)

    Parameters
    ----------
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp
    Returns
    -------
    exp : Experiment
        sorted by taxonomy
    '''
    logger.debug('sorting by taxonomies')
    taxonomy = _get_taxonomy_string(exp, remove_underscore=True)
    sort_pos = np.argsort(taxonomy, kind='mergesort')
    exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
    return exp


@Experiment._record_sig
def cluster_data(exp, axis=0, transform=None, metric='euclidean', inplace=False, **kwargs):
    '''Cluster the samples/features.

    Reorder the features/samples so that ones with similar behavior (pattern
    across samples/features) are close to each other

    Parameters
    ----------
    aixs : 0 or 1 (optional)
        0 (default) means clustering features; 1 means clustering samples
    transform : callable or ``None``
        transform the data matrix before applying clustering method
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
        With features filtered (if min_abundance is not None) and clsutered (reordered)

    '''
    if transform is None:
        transform = _log_min_transform
    data = transform(exp.data, axis=axis, **kwargs)
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
    exp : Experiment
    '''
    logger.info('sorting samples by field %s' % field)
    if axis == 0:
        x = exp.sample_metadata
    elif axis == 1:
        x = exp.feature_metadata
    idx = np.argsort(x[field], kind='mergesort')
    return exp.reorder(idx, axis=axis, inplace=inplace)


@Experiment._record_sig
def sort_by_data(exp, axis=0, subset=None, key='log_mean', inplace=False, **kwargs):
    '''Sort features based on their mean frequency.

    Sort the 2-d array by sample (axis=0) or feature (axis=0),
    depending on ``axis``. ``key`` will be applied to ``subset`` of
    each feature (axis=0) or sample (axis=1) and return a comparative value.

    Parameters
    ----------
    axis : 0 or 1
        Apply ``key`` function on row (sort by samples) (0) or column (sort by features) (1)
    subset : None or iterable of int (optional)
        Sorting by only subset of the data.
    key : callable
        a function returns a comparative value (like ``key`` in builtin ``sorted`` function).
        It is applied on each 1-d array.
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp

    Returns
    -------
    exp : Experiment
        With features sorted by mean frequency

    '''
    if subset is None:
        data_subset = exp.data
    else:
        data_subset = exp.data.take(subset, axis=1-axis)
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


def sort_obs_center_mass(exp, field=None, numeric=True, uselog=True, inplace=False):
    '''
    sort observations based on center of mass after sorting samples by field (or None not to pre sort)
    '''


def reverse_obs(exp, inplace=False):
    '''
    reverse the order of the observations
    '''


def sort_samples_by_seqs(exp, seqs, inplace=False):
    '''
    sort the samples based on the frequencies of sequences in seqs
    '''


def sort_niche(exp, field):
    '''
    sort by niches - jamie
    '''
