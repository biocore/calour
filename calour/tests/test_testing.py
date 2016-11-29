# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import pandas as pd
from os.path import join
import pandas.util.testing as pdt

import calour as ca
from calour.testing import Tests, assert_experiment_equal


class TestTesting(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.simple = ca.read(self.simple_table, self.simple_map)
        # load the complex experiment as sparse
        self.complex = ca.read(self.complex_table, self.complex_map)

    def test_assert_experiment_equal(self):
        # basic testing
        assert_experiment_equal(self.simple, self.simple)
        with self.assertRaises(AssertionError):
            assert_experiment_equal(self.simple, self.complex)

        # is copy working?
        newexp = self.simple.deepcopy()
        assert_experiment_equal(self.simple, newexp)

        # just data
        newexp = self.simple.deepcopy()
        newexp.data[2, 2] = 43
        with self.assertRaises(AssertionError):
            assert_experiment_equal(self.simple, newexp)

        # just sample metadata
        newexp = self.simple.deepcopy()
        newexp.sample_metadata['id', 0] = 42
        with self.assertRaises(AssertionError):
            assert_experiment_equal(self.simple, newexp)

        # just feature metadata
        newexp = self.simple.deepcopy()
        newexp.feature_metadata['taxonomy', 0] = '42'
        with self.assertRaises(AssertionError):
            assert_experiment_equal(self.simple, newexp)

if __name__ == "__main__":
    unittest.main()
