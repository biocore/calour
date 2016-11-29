# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy


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

    normfactor = reads / exp.data.sum(axis=axis)
    if axis == 0:
        exp.data = exp.data * normfactor[None, :]
    else:
        exp.data = exp.data * normfactor[:, None]

    return exp
