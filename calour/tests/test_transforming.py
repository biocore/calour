# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import numpy.testing as npt
import numpy as np

import calour as ca
from calour._testing import Tests


class TestTransforming(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.simple = ca.read(self.simple_table, self.simple_map)
        # load the complex experiment as sparse
        self.complex = ca.read(self.complex_table, self.complex_map)

    def test_normalize(self):
        # test normalizing to 10k reads/sample
        normed = self.simple.normalize()
        expected = np.ones([len(normed.sample_metadata)]) * 10000
        npt.assert_array_almost_equal(np.sum(normed.get_data(sparse=False, getcopy=True), axis=1), expected)

        newexp = self.complex.filter_by_data('sum_abundance', cutoff=1)
        normed = newexp.normalize()
        expected = np.ones([len(newexp.sample_metadata)]) * 10000
        out_sum = np.sum(normed.get_data(sparse=False, getcopy=True), axis=1)
        npt.assert_array_almost_equal(out_sum, expected)

if __name__ == "__main__":
    unittest.main()
