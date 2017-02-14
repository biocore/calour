# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

from skbio.util import get_data_path

from calour._testing import Tests, assert_experiment_equal
import calour as ca


class FilteringTests(Tests):
    def setUp(self):
        super().setUp()
        self.test2 = ca.read(self.test2_biom, self.test2_samp, self.test2_feat)

    def test_downsample_sample(self):
        obs = self.test2.downsample('group')
        # should be down to 4 samples; feature number is the same
        self.assertEqual(obs.shape, (4, 8))

        sid = obs.sample_metadata.index.tolist()
        all_sid = self.test2.sample_metadata.index.tolist()
        exp = self.test2.reorder([all_sid.index(i) for i in sid])
        assert_experiment_equal(obs, exp)

    def test_downsample_feature(self):
        obs = self.test2.downsample('oxygen', axis=1)
        sid = obs.feature_metadata.index.tolist()
        self.assertEqual(obs.shape, (9, 4))

        all_sid = self.test2.feature_metadata.index.tolist()
        exp = self.test2.reorder([all_sid.index(i) for i in sid], axis=1)
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
        # all features are filtered out
        obs = self.test2.filter_by_data('sum_abundance', axis=1, cutoff=10000)
        self.assertEqual(obs.shape, (9, 0))
        # none is filtered out
        obs = self.test2.filter_by_data('sum_abundance', axis=1, cutoff=1)
        assert_experiment_equal(obs, self.test2)
        # one feature is filtered out
        obs = self.test2.filter_by_data('sum_abundance', axis=1, cutoff=25)
        self.assertEqual(obs.shape, (9, 7))
        exp = ca.read(*[get_data_path(i) for i in [
            'test2.biom.filter.feature',
            'test2.sample.filter.feature',
            'test2.feature.filter.feature']])
        assert_experiment_equal(obs, exp)


if __name__ == '__main__':
    main()
