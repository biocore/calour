# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from os.path import join

import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import pandas.util.testing as pdt

import calour as ca
from calour._testing import Tests, assert_experiment_equal


class TestTransforming(Tests):
    def setUp(self):
        super().setUp()
        self.test2 = ca.read(self.test2_biom, self.test2_samp, self.test2_feat)

    def test_transform(self):
        obs = self.test2.transform()
        self.assertEqual(obs, self.test2)
        self.assertIsNot(obs, self.test2)

        obs = self.test2.transform(inplace=True)
        self.assertIs(obs, self.test2)

    def test_scale(self):
        obs = self.test2.scale()
        self.assertIsNot(obs, self.test2)

        obs = self.test2.scale(inplace=True)
        self.assertIs(obs, self.test2)

    def test_log_n(self):
        obs = self.test2.log_n()
        self.test2.data = np.log2(
            [[10., 20., 1., 20., 5., 100., 844., 100.],
             [10., 20., 2., 19., 1., 100., 849., 200.],
             [10., 20., 3., 18., 5., 100., 844., 300.],
             [10., 20., 4., 17., 1., 100., 849., 400.],
             [10., 20., 5., 16., 4., 100., 845., 500.],
             [10., 20., 6., 15., 1., 100., 849., 600.],
             [10., 20., 7., 14., 3., 100., 846., 700.],
             [10., 20., 8., 13., 1., 100., 849., 800.],
             [10., 20., 9., 12., 7., 100., 842., 900.]])
        assert_experiment_equal(obs, self.test2)
        self.assertIsNot(obs, self.test2)

        obs = self.test2.log_n(inplace=True)
        self.assertIs(obs, self.test2)

    def test_normalize(self):
        total = 1000
        obs = self.test2.normalize(total)
        assert_almost_equal(obs.data.sum(axis=1).A1,
                            [total] * 9)
        self.assertIsNot(obs, self.test2)

        obs = self.test2.normalize(total, inplace=True)
        self.assertIs(obs, self.test2)


if __name__ == '__main__':
    main()
