'''
transforming (:mod:`calour.transforming`)
=========================================

This module contains functions that transform the data table - :attr:`Experiment.data`.

.. warning:: Some of the functions require dense matrix and thus will change ``Experiment.data`` to dense matrix.

.. currentmodule:: calour.transforming

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   normalize
   normalize_by_subset_features
   normalize_compositional
   standardize
   permute_data
   binarize
   log_n
   center_log_ratio
   subsample_count
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
from sklearn import preprocessing

from . import Experiment


logger = getLogger(__name__)


def normalize(exp: Experiment, total=10000, axis=0, inplace=False) -> Experiment:
    '''Normalize the sum of each sample (axis=0) or feature (axis=1) to the same total.

    Parameters
    ----------
    total : float
        the sum (along axis) to normalize to.
    axis : {0, 1, 's', 'f'}, default=0
        the axis to normalize. 0 or 's' is normalize each sample;
        1 or 'f' to normalize each feature
    inplace : bool, default=False
        False to create a copy, True to replace values in exp

    Returns
    -------
    Experiment
        the normalized experiment
    '''
    if total <= 0:
        raise ValueError('Normalization total (%s) must be positive' % total)
    if not inplace:
        exp = deepcopy(exp)
    exp.data = preprocessing.normalize(exp.data, norm='l1', axis=1-axis) * total
    # store the normalization depth into the experiment metadata
    exp.normalized = total
    return exp


def normalize_by_subset_features(exp: Experiment, features, total=10000,
                                 negate=True, inplace=False) -> Experiment:
    '''Normalize each sample to their total sums without a list of features

    Normalizes all features (including in the exclude list) by the
    total sum calculated without the excluded features. This is to
    alleviate the compositionality in the data set by only keeping the
    features that you think are not changing across samples.

    .. note:: sum is not identical in all samples after normalization
       (since we also keep the features excluded during normalization.)

    Parameters
    ----------
    features : list-like of str
        Any container type that has ``in`` membership testing function.
        The feature IDs to exclude (or include if negate=False).
    total : int, optional
        The total abundance for the non-excluded features per sample
    negate : bool, optional
        True (default) to calculate normalization factor without features in features list.
        False to calculate normalization factor only with features in features list.
    inplace : bool, optional
        False (default) to create a new experiment, True to normalize in place

    Returns
    -------
    Experiment
        The normalized experiment

    See Also
    --------
    normalize_compositional
    '''
    feature_pos = exp.feature_metadata.index.isin(features)
    if negate:
        feature_pos = np.invert(feature_pos)
    data = exp.get_data(sparse=False)
    use_reads = np.sum(data[:, feature_pos], axis=1)
    if not inplace:
        exp = deepcopy(exp)
    # a[:, None] is the same with a[:, np.newaxis]
    exp.data = total * data / use_reads[:, None]
    # store the normalization depth into the experiment metadata
    exp.normalized = total
    return exp


def normalize_compositional(exp: Experiment, frac=0.05, total=10000, inplace=False) -> Experiment:
    '''Normalize each sample by ignoring the features with mean>=frac in all the experiment

    This assumes that the majority of features have mean abundance
    less than a certain fraction; and that the majority of features don't
    change across samples in a constant direction. Thus, this function
    select out these features and use their sum across samples for normalization.

    Parameters
    ----------
    frac : float, optional
        ignore features with mean (over all samples) >= frac.
    total : int, optional
        The total abundance for the non-excluded features per sample
    inplace : bool, optional
        False (default) to create a new experiment, True to normalize in place

    Returns
    -------
    Experiment
        The normalized experiment.

    See Also
    --------
    normalize_by_subset_features

    '''
    comp_features = exp.filter_mean_abundance(frac)
    logger.info('ignoring %d features' % comp_features.shape[1])
    newexp = exp.normalize_by_subset_features(comp_features.feature_metadata.index.values,
                                              total=total, negate=True, inplace=inplace)
    return newexp


def rescale(exp: Experiment, total=10000, axis=0, inplace=False) -> Experiment:
    '''Rescale the data to mean sum of all samples (axis=0) or features (axis=1) to be total.

    This function rescales by multiplying ALL entries in :attr:`.Experiment.data` by same number.

    Parameters
    ----------
    total : float
        The value that the mean sum (along axis) will be equal to after rescaling.
    axis : 0, 1, 's', or 'f', optional
        the axis to normalize. 0 or 's' (default) is normalize each sample;
        1 or 'f' to normalize each feature
    inplace : bool, optional
        False (default) to create a copy, True to replace values in exp

    Returns
    -------
    Experiment

    '''
    if not inplace:
        exp = deepcopy(exp)
    current_mean = np.mean(exp.data.sum(axis=1-axis))
    exp.data = exp.data * total / current_mean
    return exp


def standardize(exp: Experiment, axis=0, inplace=False) -> Experiment:
    '''Standardize a dataset along an axis.

    This transforms the data into zero mean and unit variance. It
    calls :func:`sklearn.preprocessing.scale` to do the real work.

    .. warning:: It will convert the ``Experiment.data`` from the sparse matrix to dense array.

    Parameters
    ----------
    axis : 0, 1, 's', or 'f'
        0 or 's'  means scaling occurs sample-wise; 1 or 'f' feature-wise.

    Returns
    -------
    Experiment

    '''
    logger.debug('scaling the data, axis=%d' % axis)
    if not inplace:
        exp = deepcopy(exp)
    if exp.sparse:
        exp.sparse = False
    preprocessing.scale(exp.data, axis=1-axis, copy=False)
    return exp


def binarize(exp: Experiment, threshold=1, inplace=False) -> Experiment:
    '''Binarize the data with a threshold.

    It calls :func:`sklearn.preprocessing.binarize` to do the real work.

    Parameters
    ----------
    threshold : Numeric
        the cutoff value. Any values below or equal to this will be replaced by 0;
        values above it by 1.
    '''
    logger.debug('binarizing the data with threshold=%f' % threshold)
    if not inplace:
        exp = deepcopy(exp)
    preprocessing.binarize(exp.data, threshold=threshold, copy=False)
    return exp


def log_n(exp: Experiment, n=1, inplace=False) -> Experiment:
    '''Log transform the data.

    Parameters
    ----------
    n : numeric, optional
        cap the tiny values (any value smaller than ``n`` will be replaced by ``n``)
        and then log transform the data.

    '''
    logger.debug('log_n transforming the data, min. threshold=%f' % n)
    if not inplace:
        exp = deepcopy(exp)

    if exp.sparse:
        exp.sparse = False

    exp.data[exp.data < n] = n
    exp.data = np.log2(exp.data)

    return exp


def permute_data(exp: Experiment, normalize=True, inplace=False, random_seed=None) -> Experiment:
    '''Shuffle independently the abundances of each feature.

    This creates a new experiment with no dependency between features.

    Parameters
    ----------
    normalize : bool, optional
        True (default) to normalize each sample after completing the feature shuffling.
        False to not normalize
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed for the random permutations
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    Returns
    -------
    Experiment
        With each feature shuffled independently

    '''
    # create the numpy.random.Generator
    rng = np.random.default_rng(random_seed)

    if not inplace:
        exp = deepcopy(exp)

    exp.sparse = False
    for cfeature in range(exp.shape[1]):
        rng.shuffle(exp.data[:, cfeature])
    if normalize:
        exp.normalize(np.mean(exp.data.sum(axis=1)), inplace=True)
    return exp


def center_log_ratio(exp: Experiment, method=lambda matrix: matrix + 1, centralize=False, inplace=False):
    """ Performs a clr transform to each sample.

    Parameters
    ----------
    method : callable, optional
        An optional function to specify how the pseudocount method should be
        handled (to deal with zeros in the matrix)
    centralize : bool, optional
        centralize feature-wise to zero or not
    inplace : bool, optional
        False (default) to create a new experiment, True to normalize in place

    Returns
    -------
    Experiment
        The normalized experiment. Note that all features are clr normalized.

    See Also
    --------
    skbio.stats.composition.clr
    skbio.stats.composition.centralize
    """
    from skbio.stats.composition import clr, centralize as skbio_centralize

    logger.debug('clr transforming the data')
    if not inplace:
        exp = deepcopy(exp)
    if exp.sparse:
        exp.sparse = False
    if centralize:
        exp.data = clr(skbio_centralize(method(exp.data)))
    else:
        exp.data = clr(method(exp.data))
    return exp


def subsample_count(exp: Experiment, total, replace=False, inplace=False, random_seed=None):
    """Randomly subsample each sample to the same number of counts.

    .. warning:: This function will change the :attr:`Experiment.data`
       object from sparse to dense. The input ``Experiment`` object
       should not have been normalized by total sum and its data
       should be discrete count. The samples that have few total count
       than ``total`` will be dropped.

    .. note:: This function may not work on Windows OS. It relies on
       the :func:`skbio.stats.subsample_counts` which have
       `ValueError: Buffer dtype mismatch, expected 'int64_t' but got
       'long'` in `_subsample_counts_without_replacement` function of
       `skbio/stats/__subsample.pyx`

    Parameters
    ----------
    total : int, optional
        cap the tiny values and then clr transform the data.
    replace : bool, optional
        If True, subsample with replacement. If False (the default), subsample without replacement
    inplace : bool, optional
        False (default) to create a new experiment, True to do it in place
    random_seed : int or None, optional, default=None
        passed to :func:`numpy.random.seed`

    Returns
    -------
    Experiment
        The subsampled experiment.

    See Also
    --------
    :func:`skbio.stats.subsample_counts`

    """
    # import here to make skbio optional dependency
    from skbio.stats import subsample_counts

    if not inplace:
        exp = deepcopy(exp)
    if exp.sparse:
        exp.sparse = False
    # subsample_counts() require int as input; if not, raise error
    if exp.data.dtype.kind not in {'u', 'i'}:
        raise ValueError('Your `Experiment` object is normalized: subsample operates on integer raw data, not on normalized data.')

    drops = []
    np.random.seed(random_seed)
    for row in range(exp.data.shape[0]):
        counts = exp.data[row, :]
        if total > counts.sum() and not replace:
            drops.append(row)
        else:
            exp.data[row, :] = subsample_counts(counts, n=total, replace=replace)

    exp.reorder([i not in drops for i in range(exp.data.shape[0])], inplace=True)
    exp.normalized = total
    return exp
