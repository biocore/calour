# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy

import scipy
import numpy as np
from sklearn import preprocessing

from .filtering import _filter_by_data

logger = getLogger(__name__)


def normalize(exp, reads=10000, axis=1, inplace=False):
    '''Normalize the sum of each sample (axis=0) or feature (axis=1) to sum reads

    Parameters
    ----------
    exp : Experiment
    reads : float
        the sum (along axis) to normalize to
    axis : int (optional)
        the axis to normalize. 1 (default) is normalize each sample, 0 to normalize each feature
    inplace : bool (optional)
        False (default) to create a copy, True to replace values in exp

    Returns
    -------
    newexp : Experiment
        the normalized experiment
    '''
    if not inplace:
        exp = deepcopy(exp)
    exp.data = preprocessing.normalize(exp.data, 'l1', axis=axis, copy=False) * reads

    return exp


def normalize_filter_features(exp, features, reads=10000, exclude=True, inplace=False):
    '''Normalize the sum of each sample without a list of features

    Normalizes all features (including in the exclude list) after calulcating the scaling
    without the excluded features.
    Note: sum is not identical in all samples after normalization (since also keeps the
    excluded features)

    Parameters
    ----------
    features : list of str
        The features to exclude (or include if exclude=False)
    reads : int (optional)
        The number of reads for the non-excluded features per sample
    exclude : bool (optional)
        True (default) to calculate normalization factor without features in features list.
        False to calculate normalization factor only with features in features list.
    inplace : bool (optional)
        False (default) to create a new experiment, True to normalize in place

    Returns
    -------
    newexp : calour.Experiment
        The normalized experiment
    '''
    feature_pos = exp.feature_metadata.index.isin(features)
    if exclude:
        feature_pos = np.invert(feature_pos)
    data = exp.get_data(sparse=False)
    use_reads = np.sum(data[:, feature_pos], axis=1)
    if inplace:
        newexp = exp
    else:
        newexp = deepcopy(exp)
    newexp.data = reads * data / use_reads[:, None]
    return newexp


def _log_min_transform(data, axis=1, min_abundance=None, logit=1, normalize=True):
    '''Transform and normalize the data.
    Operation is done on features or samples (depending on axis)
    Filtering/normalization steps are:
    minimal total reads - remove all features (axis=1) or samples (axis=0) with <min_abundance total
    log transform - using a minimal value (all entries < logit are transformed to logit before the log)
    normalize - normalize each feature (axis=1) or sample (axis=0) to mean=0 std=1

    Parameters
    ----------
    data : 2d nparray or scipy.sparse
        The data to transform. If sparse, it is converted to dense
    axis : int (optional)
        axis=1 normalizes the features, axis=0 normalizes the samples
    min_abundance : None or float (optional)
        None (default) to not remove any features.
        float to remove all features with total reads < float
    logit : float or None (optional)
        float (default) to log transform the data before clustering, using logit as the minimal threshold
        (data<logit is changed to logit)
        None to not log transform.
    normalize : bool (optional)
        True (default) to normalize each feature to sum 1 std 1.
        False to not normalize each feature.

    Returns
    -------
    ndarray
        transformed 2-d array
    '''
    if scipy.sparse.issparse(data):
        new = data.toarray()
    else:
        new = data.copy()

    # filter low-freq rows/columns
    if min_abundance is not None:
        logger.debug('filtering min abundance %d' % min_abundance)
        select = _filter_by_data(
            data, 'sum_abundance', axis=axis, cutoff=min_abundance)
        new = np.take(new, np.where(select)[0], axis=axis)

    if logit is not None:
        new[new < logit] = logit
        new = np.log2(new)

    if normalize is True:
        # center and normalize
        new = preprocessing.scale(new, axis=axis, copy=False)
    return new
