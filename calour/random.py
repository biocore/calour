'''
random (:mod:`calour.random`)
==============================

.. currentmodule:: calour.random

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   set_random_seed
   choice
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from logging import getLogger

from numpy.random import RandomState


logger = getLogger(__name__)

# this is the global random seed; if it's None, it is unset
random_state = RandomState()


def set_random_seed(seed):
    '''Set the calour specific RandomState seed

    Parameters
    ----------
    seed : int or None
        see numpy.random.RandomState()
    '''
    global random_state

    logger.debug('seeding random state with value %r' % seed)
    random_state = RandomState(seed)


def choice(*kargs, **kwargs):
    return random_state.choice(*kargs, **kwargs)
