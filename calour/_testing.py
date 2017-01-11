# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase
from os.path import join, dirname, abspath

import pandas.util.testing as pdt
import numpy.testing as npt

import calour as ca


class Tests(TestCase):
    def setUp(self):
        test_data_dir = join(dirname(abspath(__file__)), 'tests', 'data')
        self.test_data_dir = test_data_dir
        self.simple_table = join(test_data_dir, 'test1.biom')
        self.simple_map = join(test_data_dir, 'test1.map.txt')
        self.complex_table = join(test_data_dir, 'timeseries.biom')
        self.complex_map = join(test_data_dir, 'timeseries.map.txt')


def assertIsInstance(obj, cls, msg=''):
    """Test that obj is an instance of cls
    (which can be a class or a tuple of classes,
    as supported by isinstance()).
    (copied From Pandas unit testing module)"""
    if not isinstance(obj, cls):
        err_msg = "{0}Expected type {1}, found {2} instead"
        raise AssertionError(err_msg.format(msg, cls, type(obj)))


def assert_experiment_equal(exp1, exp2, check_history=False, almost_equal=True):
    '''Test if two experiments are equal

    Parameters
    ----------
    exp1 : Experiment
    exp2 : Experiment
    check_history : bool (optional)
        False (default) to skip testing the command history, True to compare also the command history
    almost_equal : bool (optional)
        True (default) to test for almost identical, False to test the data matrix for exact identity
    '''
    assertIsInstance(exp1, ca.Experiment, 'exp1 not a calour Experiment class')
    assertIsInstance(exp2, ca.Experiment, 'exp2 not a calour Experiment class')

    pdt.assert_frame_equal(exp1.feature_metadata, exp2.feature_metadata)
    pdt.assert_frame_equal(exp1.sample_metadata, exp2.sample_metadata)
    if almost_equal:
        dat1 = exp1.get_data(sparse=False, getcopy=True)
        dat2 = exp2.get_data(sparse=False, getcopy=True)
        npt.assert_array_almost_equal(dat1, dat2)
    else:
        npt.assert_array_equal(exp1.data, exp2.data)
    if check_history:
        if not exp1._call_history == exp2._call_history:
            raise AssertionError('histories are different between exp1 and exp2')
