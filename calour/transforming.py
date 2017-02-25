# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy
from collections import defaultdict

import numpy as np
from sklearn import preprocessing


logger = getLogger(__name__)


def normalize(exp, total=10000, axis=1, inplace=False):
    '''Normalize the sum of each sample (axis=0) or feature (axis=1) to sum total

    Parameters
    ----------
    exp : Experiment
    total : float
        the sum (along axis) to normalize to
    axis : int (optional)
        the axis to normalize. 1 (default) is normalize each sample, 0 to normalize each feature
    inplace : bool (optional)
        False (default) to create a copy, True to replace values in exp

    Returns
    -------
    ``Experiment``
        the normalized experiment
    '''
    if not inplace:
        exp = deepcopy(exp)
    exp.data = preprocessing.normalize(exp.data, norm='l1', axis=axis) * total
    return exp


def rescale(exp, total=10000, axis=1, inplace=False):
    '''Rescale the data to mean sum of all samples (axis=1) or features (axis=0) to be total.

    This function rescales by multiplying ALL entries in exp.data by same number.

    Parameters
    ----------
    exp : Experiment
    total : float
        the mean sum (along axis) to normalize to
    axis : int (optional)
        the axis to normalize. 1 (default) is normalize each sample, 0 to normalize each feature
    inplace : bool (optional)
        False (default) to create a copy, True to replace values in exp

    Returns
    -------
    ``Experiment``
        the normalized experiment
    '''
    if not inplace:
        exp = deepcopy(exp)
    current_mean = np.mean(exp.data.sum(axis=axis))
    exp.data = exp.data * total / current_mean
    return exp


def scale(exp, axis=1, inplace=False):
    '''Standardize a dataset along an axis

    .. warning:: It will convert the ``Experiment.data`` from the sparse matrix to dense array.

    Parameters
    ----------
    axis : 0 or 1
        1 means scaling occur sample-wise; 0 feature-wise.

    Returns
    -------
    ``Experiment``
    '''
    logger.debug('scaling the data, axis=%d' % axis)
    if not inplace:
        exp = deepcopy(exp)
    if exp.sparse:
        exp.sparse = False
    preprocessing.scale(exp.data, axis=axis, copy=False)
    return exp


def binarize(exp, threshold=1, inplace=False):
    '''Binarize the data with a threshold.

    It calls scikit-learn to do the real work.

    Parameters
    ----------
    threshold : Numeric
        the cutoff value. Any values below or equal to this will be replaced by 0,
        above it by 1.

    Returns
    -------
    ``Experiment``
    '''
    logger.debug('binarizing the data. threshold=%f' % threshold)
    if not inplace:
        exp = deepcopy(exp)
    preprocessing.binarize(exp.data, threshold=threshold, copy=False)
    return exp


def log_n(exp, n=1, inplace=False):
    '''Log transform the data

    Parameters
    ----------
    n : numeric, optional
        cap the tiny values and then log transform the data.
    inplace : bool, optional

    Returns
    -------
    ``Experiment``
    '''
    logger.debug('log_n transforming the data, min. threshold=%f' % n)
    if not inplace:
        exp = deepcopy(exp)

    if exp.sparse:
        exp.sparse = False

    exp.data[exp.data < n] = n
    exp.data = np.log2(exp.data)

    return exp


def transform(exp, steps=[], inplace=False, **kwargs):
    '''Chain transformations together.

    Parameters
    ----------
    steps : list of callable
        each callable is a transformer that takes ``Experiment`` object as
        its 1st argument and has a boolean parameter of ``inplace``. Each
        callable should return an ``Experiment`` object.
    inplace : bool
        transformation occuring in the original data or a copy
    kwargs : dict
        keyword arguments to pass to each transformers. The key should
        be in the form of "<transformer_name>__<param_name>". For
        example, "transform(exp, steps=[log_n], log_n__n=3)" will set
        "n" of function "log_n" to 3

    Returns
    -------
    ``Experiment``
        with its data transformed

    '''
    if not inplace:
        exp = deepcopy(exp)
    params = defaultdict(dict)
    for k, v in kwargs.items():
        transformer, param_name = k.split('__')
        if param_name == 'inplace':
            raise ValueError('You should not give %s argument. It should be '
                             'set thru `inplace` argument for this function.')
        params[transformer][param_name] = v
    for step in steps:
        step(exp, inplace=True, **params[step.__name__])
    return exp


def normalize_by_subset_features(exp, features, total=10000, exclude=True, inplace=False):
    '''Normalize each sample by their total sums without a list of features

    Normalizes all features (including in the exclude list) by the
    total sum calculated without the excluded features. This is to
    alleviate the compositionality in the data set by only keeping the
    features that you think are not changing across samples.

    .. note:: sum is not identical in all samples after normalization
       (since also keeps the excluded features)

    Parameters
    ----------
    features : list of str
        The features to exclude (or include if exclude=False)
    total : int (optional)
        The total abundance for the non-excluded features per sample
    exclude : bool (optional)
        True (default) to calculate normalization factor without features in features list.
        False to calculate normalization factor only with features in features list.
    inplace : bool (optional)
        False (default) to create a new experiment, True to normalize in place

    Returns
    -------
    ``Experiment``
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
    newexp.data = total * data / use_reads[:, None]
    return newexp
