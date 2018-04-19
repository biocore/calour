'''
transforming (:mod:`calour.transforming`)
=========================================

.. warning:: Some of the functions require dense matrix and thus will change the sparse matrix to dense matrix.

.. currentmodule:: calour.transforming

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   normalize
   normalize_by_subset_features
   normalize_compositional
   scale
   random_permute_data
   binarize
   log_n
   transform
   center_log
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
from collections import defaultdict

import numpy as np
from sklearn import preprocessing
from skbio.stats.composition import clr, centralize as skbio_centralize
from skbio.stats import subsample_counts

from . import Experiment
from ._doc import ds


logger = getLogger(__name__)


@Experiment._record_sig
def normalize(exp: Experiment, total=10000, axis=0, inplace=False):
    '''Normalize the sum of each sample (axis=0) or feature (axis=1) to sum total

    Parameters
    ----------
    total : float
        the sum (along axis) to normalize to
    axis : 0, 1, 's', or 'f', optional
        the axis to normalize. 0 or 's' (default) is normalize each sample;
        1 or 'f' to normalize each feature
    inplace : bool, optional
        False (default) to create a copy, True to replace values in exp

    Returns
    -------
    Experiment
        the normalized experiment
    '''
    if isinstance(total, bool):
        raise ValueError('Normalization total (%s) not numeric' % total)
    if total <= 0:
        raise ValueError('Normalization total (%s) must be positive' % total)
    if not inplace:
        exp = deepcopy(exp)
    exp.data = preprocessing.normalize(exp.data, norm='l1', axis=1-axis) * total
    # store the normalization depth into the experiment metadata
    exp.exp_metadata['normalized'] = total
    return exp


@Experiment._record_sig
def rescale(exp: Experiment, total=10000, axis=0, inplace=False):
    '''Rescale the data to mean sum of all samples (axis=0) or features (axis=1) to be total.

    This function rescales by multiplying ALL entries in exp.data by same number.

    Parameters
    ----------
    total : float
        the mean sum (along axis) to normalize to
    axis : 0, 1, 's', or 'f', optional
        the axis to normalize. 0 or 's' (default) is normalize each sample;
        1 or 'f' to normalize each feature
    inplace : bool, optional
        False (default) to create a copy, True to replace values in exp

    Returns
    -------
    Experiment
        the normalized experiment
    '''
    if not inplace:
        exp = deepcopy(exp)
    current_mean = np.mean(exp.data.sum(axis=1-axis))
    exp.data = exp.data * total / current_mean
    return exp


@Experiment._record_sig
def scale(exp: Experiment, axis=0, inplace=False):
    '''Standardize a dataset along an axis

    .. warning:: It will convert the ``Experiment.data`` from the sparse matrix to dense array.

    Parameters
    ----------
    axis : 0, 1, 's', or 'f'
        0 or 's'  means scaling occur sample-wise; 1 or 'f' feature-wise.

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


@Experiment._record_sig
def binarize(exp: Experiment, threshold=1, inplace=False):
    '''Binarize the data with a threshold.

    It calls scikit-learn to do the real work.

    Parameters
    ----------
    threshold : Numeric
        the cutoff value. Any values below or equal to this will be replaced by 0,
        above it by 1.

    Returns
    -------
    Experiment
    '''
    logger.debug('binarizing the data. threshold=%f' % threshold)
    if not inplace:
        exp = deepcopy(exp)
    preprocessing.binarize(exp.data, threshold=threshold, copy=False)
    return exp


@Experiment._record_sig
def log_n(exp: Experiment, n=1, inplace=False):
    '''Log transform the data

    Parameters
    ----------
    n : numeric, optional
        cap the tiny values and then log transform the data.
    inplace : bool, optional

    Returns
    -------
    Experiment
    '''
    logger.debug('log_n transforming the data, min. threshold=%f' % n)
    if not inplace:
        exp = deepcopy(exp)

    if exp.sparse:
        exp.sparse = False

    exp.data[exp.data < n] = n
    exp.data = np.log2(exp.data)

    return exp


@ds.get_sectionsf('transforming.transform')
@Experiment._record_sig
def transform(exp: Experiment, steps=[], inplace=False, **kwargs):
    '''Chain transformations together.

    Parameters
    ----------
    steps : list of callable
        each callable is a transformer that takes :class:`.Experiment` object as
        its 1st argument and has a boolean parameter of ``inplace``. Each
        callable should return an :class:`.Experiment` object.
    inplace : bool
        transformation occuring in the original data or a copy
    kwargs : dict
        keyword arguments to pass to each transformers. The key should
        be in the form of "<transformer_name>__<param_name>". For
        example, "transform(exp: Experiment, steps=[log_n], log_n__n=3)" will set
        "n" of function "log_n" to 3

    Returns
    -------
    Experiment
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


@Experiment._record_sig
def normalize_by_subset_features(exp: Experiment, features, total=10000, negate=True, inplace=False):
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
        The feature IDs to exclude (or include if negate=False)
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
    '''
    feature_pos = exp.feature_metadata.index.isin(features)
    if negate:
        feature_pos = np.invert(feature_pos)
    data = exp.get_data(sparse=False)
    use_reads = np.sum(data[:, feature_pos], axis=1)
    if inplace:
        newexp = exp
    else:
        newexp = deepcopy(exp)
    newexp.data = total * data / use_reads[:, None]
    # store the normalization depth into the experiment metadata
    newexp.exp_metadata['normalized'] = total
    return newexp


def normalize_compositional(exp: Experiment, min_frac=0.05, total=10000, inplace=False):
    '''Normalize each sample by ignoring the features with mean>=min_frac in all the experiment

    This assumes that the majority of features have less than min_frac mean, and that the majority of features don't change
    between samples in a constant direction

    Parameters
    ----------
    min_frac : float, optional
        ignore features with mean (over all samples) >= min_frac.
    total : int, optional
        The total abundance for the non-excluded features per sample
    inplace : bool, optional
        False (default) to create a new experiment, True to normalize in place

    Returns
    -------
    Experiment
        The normalized experiment. Note that all features are normalized (including the ones with mean>=min_frac)
    '''
    comp_features = exp.filter_mean(min_frac)
    logger.info('ignoring %d features' % comp_features.shape[1])
    newexp = exp.normalize_by_subset_features(comp_features.feature_metadata.index.values,
                                              total=total, negate=True, inplace=inplace)
    return newexp


def random_permute_data(exp: Experiment, normalize=True):
    '''Shuffle independently the reads of each feature

    Creates a new experiment with no dependence between the features.

    Parameters
    ----------
    normalize : bool, optional
        True (default) to normalize each sample after completing the feature shuffling.
        False to not normalize

    Returns
    -------
    Experiment
        With each feature shuffled independently

    '''
    newexp = exp.copy()
    newexp.sparse = False
    for cfeature in range(newexp.shape[1]):
        np.random.shuffle(newexp.data[:, cfeature])
    if normalize:
        newexp.normalize(np.mean(exp.data.sum(axis=1)), inplace=True)
    return newexp


@Experiment._record_sig
def center_log(exp: Experiment, method=lambda matrix: matrix + 1, centralize=False, inplace=False):
    """ Performs a clr transform to normalize each sample.

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


@Experiment._record_sig
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
    if inplace:
        newexp = exp
    else:
        newexp = deepcopy(exp)
    if newexp.sparse:
        newexp.sparse = False
    # subsample_counts() require int as input;
    # check if it is normalized: if so, raise error
    if exp.exp_metadata.get('normalized'):
        raise ValueError('Your `Experiment` object is normalized: subsample operates on integer raw data, not on normalized data.')

    drops = []
    np.random.seed(random_seed)
    for row in range(newexp.data.shape[0]):
        counts = newexp.data[row, :]
        if total > counts.sum() and not replace:
            drops.append(row)
        else:
            newexp.data[row, :] = subsample_counts(counts, n=total, replace=replace)

    newexp.reorder([i not in drops for i in range(newexp.data.shape[0])], inplace=True)
    return newexp
