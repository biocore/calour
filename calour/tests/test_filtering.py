# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

from skbio.util import get_data_path

from calour._testing import Tests, assert_experiment_equal
import calour as ca


class FilteringTests(Tests):
    def setUp(self):
        super().setUp()
        self.exp1 = ca.read(*[get_data_path(i) for i in [
            'filter.1.biom', 'filter.1_sample.txt']])
        self.exp2 = ca.read(*[get_data_path(i) for i in [
            'filter.1.biom', 'filter.1_sample.txt']], sparse=False)

    def test_down_sample(self):
        obs = self.exp1.down_sample('group')
        sid = obs.sample_metadata.index.tolist()
        all_sid = self.exp1.sample_metadata.index.tolist()
        exp = self.exp1.reorder([all_sid.index(i) for i in sid])
        self.assertEqual(obs, exp)

    def test_filter_by_metadata_sample(self):
        obs = self.exp1.filter_by_metadata('group', 1)
        exp = ca.read(*[get_data_path(i) for i in [
            'filter.2.biom', 'filter.2_sample.txt']])
        self.assertIsNot(obs, exp)
        assert_experiment_equal(obs, exp)

        obs = self.exp1.filter_by_metadata('group', 1, inplace=True)
        self.assertIs(obs, self.exp1)


    def test_filter_by_metadata_feature(self):
        obs = self.exp1.filter_by_metadata('taxonomy', 'bad_bacteria', negate=True, axis=1)
        self.assertIsNot(obs, self.exp1)
        exp = ca.read(*[get_data_path(i) for i in ['filter.3.biom', 'filter.1_sample.txt']])
        assert_experiment_equal(obs, exp)

        obs = self.exp1.filter_by_metadata(
            'taxonomy', 'bad_bacteria', axis=1, negate=True, inplace=True)
        self.assertIs(obs, self.exp1)

    def test_filter_by_data_sample(self):
        for e in [self.exp1, self.exp2]:
            # all samples are filtered out
            obs = e.filter_by_data('sum_abundance', cutoff=10000)
            self.assertEqual(obs.data.shape[0], 0)

            obs = e.filter_by_data('sum_abundance', cutoff=1200)
            exp = ca.read(*[get_data_path(i) for i in ['filter.5.biom', 'filter.5_sample.txt']])
            assert_experiment_equal(obs, exp)

    def test_filter_by_data_feature(self):
        # test on both dense and sparse
        for e in [self.exp1, self.exp2]:
            # all features are filtered out
            obs = e.filter_by_data('sum_abundance', axis=1, cutoff=10000)
            self.assertEqual(obs.data.shape[1], 0)

            obs = e.filter_by_data('sum_abundance', axis=1, cutoff=1)
            exp = ca.read(*[get_data_path(i) for i in ['filter.4.biom', 'filter.4_sample.txt']])
            assert_experiment_equal(obs, exp)


if __name__ == '__main__':
    main()
