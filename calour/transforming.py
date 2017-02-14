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


logger = getLogger(__name__)


def normalize(exp, total=10000, axis=1, inplace=False):
    '''Normalize the sum of each sample (axis=0) or feature (axis=1) to sum reads

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


def scale(exp, axis=1, inplace=False):
    '''Standardize a dataset along an axis

    .. warning:: It will convert the sparse matrix to dense array.
    '''
    if not inplace:
        exp = deepcopy(exp)
    if exp.sparse:
        exp.sparse = False
    preprocessing.scale(exp.data, axis=axis, copy=False)
    return exp


def log_n(exp, n=1, inplace=False):
    '''Log transform the data

    Parameters
    ----------
    n : numeric, optional
        cap the tiny values and then log transform the data.
    inplace : bool, optional
    '''
    if not inplace:
        exp = deepcopy(exp)

    if exp.sparse:
        exp.sparse = False

    exp.data[exp.data < n] = n
    exp.data = np.log2(exp.data)

    return exp


def transform(exp, steps=[], inplace=False):
    '''Chain transformations together.'''
    if not inplace:
        exp = deepcopy(exp)
    for step in steps:
        step(exp, inplace=True)
    return exp
