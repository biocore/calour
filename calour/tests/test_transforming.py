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
from calour.transforming import _log_min_transform, normalize_filter_features
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

    def test_log_min_transform(self):
        data = self.simple.data
        # test axis=1, with min_abundance, log transform and no normalization
        ndata = _log_min_transform(data, axis=1, min_abundance=1000, logit=10, normalize=False)
        # test we didn't filter sample
        self.assertEqual(ndata.shape[0], data.shape[0])
        # test we did remove the features
        self.assertEqual(ndata.shape[1], 3)
        # test we log transformed the data
        self.assertEqual(ndata[0, 0], np.log2(100))
        self.assertEqual(ndata[3, 2], np.log2(400))

        # test axis=1, without min_abundance, no log transform and with normalization
        ndata = _log_min_transform(data, axis=1, min_abundance=None, logit=None, normalize=True)
        # test we didn't filter samples
        self.assertEqual(ndata.shape[0], data.shape[0])
        # test we didn't remove features
        self.assertEqual(ndata.shape[1], data.shape[1])
        # test we log normalized the data to mean 0, std 1
        npt.assert_array_almost_equal(np.mean(ndata, axis=1), np.zeros([ndata.shape[0]]))
        npt.assert_array_almost_equal(np.std(ndata, axis=1), np.ones([ndata.shape[0]]))

    def test_normalize_filter_features(self):
        # test the filtering in standard mode (remove a few features, normalize to 10k)
        exp = self.simple
        bad_features = [6, 7]
        features = [exp.feature_metadata.index[cbad] for cbad in bad_features]
        newexp = normalize_filter_features(exp, features, reads=10000, exclude=True, inplace=False)
        # see the mean of the features we want (without 6,7) is 10k
        good_features = list(set(range(exp.data.shape[1])).difference(set(bad_features)))
        npt.assert_array_almost_equal(newexp.data[:, good_features].sum(axis=1), np.ones([exp.data.shape[0]])*10000)
        self.assertTrue(np.all(newexp.data[:, bad_features] > exp.data[:, bad_features]))


if __name__ == "__main__":
    unittest.main()
