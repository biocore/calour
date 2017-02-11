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


def _log_min_transform(data, axis=1, min_abundance=None, logit=1, normalize=True):
    '''transform the data array.

    Parameters
    ----------
    min_abundance : None or float (optional)
        None (default) to not remove any features.
        float to remove all features with total reads < float (to make clustering faster).
    lgoit : bool (optional)
        True (default) to log transform the data before clustering.
        False to not log transform.
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
            'sum_abundance', axis=axis, cutoff=min_abundance)
        new = np.take(new, select, axis=axis)

    if logit is not None:
        new[new < logit] = logit
        new = np.log2(new)

    if normalize is True:
        # center and normalize
        new = preprocessing.scale(new, axis=axis, copy=False)

    return new
