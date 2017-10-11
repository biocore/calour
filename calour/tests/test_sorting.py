# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from os.path import join

import pandas.util.testing as pdt
import numpy as np

import calour as ca
from calour._testing import Tests, assert_experiment_equal


class SortingTests(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.test2 = ca.read(self.test2_biom, self.test2_samp, self.test2_feat, normalize=None)
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        # load the complex experiment as sparse
        self.timeseries = ca.read(self.timeseries_biom, self.timeseries_samp, normalize=None)

    def test_cluster_data(self):
        def log_and_scale(exp):
            exp.log_n(inplace=True)
            exp.scale(inplace=True, axis=1)
            return exp
        # no minimal filtering
        clustered = self.test1.cluster_data(transform=log_and_scale, axis=0)
        new_ids = ['badsample', 'S20', 'S18', 'S16', 'S12', 'S14', 'S19', 'S17', 'S13',
                   'S15', 'S1', 'S9', 'S11', 'S3', 'S5', 'S7', 'S2', 'S4', 'S6', 'S8',
                   'S10']
        self.assertListEqual(new_ids, clustered.sample_metadata.index.tolist())

    def test_cluster_features(self):
        exp = self.test1.cluster_features()
        new_ids = ['AC', 'AT', 'AA', 'TA', 'TG', 'TC', 'AG', 'GG', 'TT', 'GT', 'GA', 'badfeature']
        self.assertListEqual(exp.feature_metadata.index.tolist(), new_ids)

    def test_sort_by_metadata_sample(self):
        # test sorting various fields (keeping the order)
        obs = self.timeseries.sort_by_metadata(
            field='MINUTES', inplace=True).sort_by_metadata(
                field='HOUR', inplace=True).sort_by_metadata(
                    field='DAY', inplace=True)
        self.assertIs(obs, self.timeseries)
        exp = ca.read(join(self.test_data_dir, 'timeseries.sorted.time.biom'),
                      join(self.test_data_dir, 'timeseries.sample'),
                      normalize=None)
        assert_experiment_equal(obs, exp, almost_equal=True)
        self.assertListEqual(obs.sample_metadata['MF_SAMPLE_NUMBER'].tolist(), list(range(1, 96)))

    def test_sort_samples(self):
        obs = self.timeseries.sort_samples('MINUTES', inplace=True).sort_samples(
                'HOUR', inplace=True).sort_samples(
                    'DAY', inplace=True)
        self.assertIs(obs, self.timeseries)
        exp = ca.read(join(self.test_data_dir, 'timeseries.sorted.time.biom'),
                      join(self.test_data_dir, 'timeseries.sample'),
                      normalize=None)
        assert_experiment_equal(obs, exp, almost_equal=True)
        self.assertListEqual(obs.sample_metadata['MF_SAMPLE_NUMBER'].tolist(), list(range(1, 96)))

    def test_sort_by_metadata_feature(self):
        obs = self.test2.sort_by_metadata(
            field='level2', axis=1).sort_by_metadata(
                field='level1', axis=1)
        self.assertIsNot(obs, self.test2)
        assert_experiment_equal(
            obs, self.test2.reorder(obs.feature_metadata['ori.order'], axis=1))
        self.assertListEqual(obs.feature_metadata['new.order'].tolist(), list(range(8)))

    def test_sort_by_data_sample(self):
        # sort sample based on the first and last features
        obs = self.test2.sort_by_data(subset=[0, 7])
        # the order is the same with original
        assert_experiment_equal(obs, self.test2)

        obs = self.test2.sort_by_data(subset=[0, 3])
        assert_experiment_equal(
            obs, self.test2.reorder(obs.sample_metadata['ori.order'], axis=0))
        self.assertListEqual(obs.sample_metadata['new.order'].tolist(), list(range(9)))

    def test_sort_by_data_feature(self):
        obs = self.timeseries.sort_by_data(axis=1)
        exp = ca.read(join(self.test_data_dir, 'timeseries.sorted.freq.biom'),
                      join(self.test_data_dir, 'timeseries.sample'),
                      normalize=None)
        assert_experiment_equal(obs, exp, almost_equal=True)

    def test_sort_centroid(self):
        exp = self.test1.sort_centroid()
        # we didn't change the samples
        pdt.assert_frame_equal(exp.sample_metadata, self.test1.sample_metadata)
        # we did change the features but only reordered
        self.assertEqual(set(exp.feature_metadata.index), set(self.test1.feature_metadata.index))
        # look if one feature is in the correct place
        self.assertEqual(exp.feature_metadata.index[1], self.test1.feature_metadata.index[4])

    def test_sort_abundance_mean(self):
        exp = self.test1.sort_abundance(key=np.mean)
        new_ids = ['GA', 'GT', 'badfeature', 'TT', 'AC', 'AA', 'AG', 'TA', 'AT', 'TG', 'TC', 'GG']
        self.assertEqual(exp.feature_metadata.index.tolist(), new_ids)
        self.assertEqual(exp.shape, self.test1.shape)

    def test_sort_abundance_default(self):
        exp = self.test1.sort_abundance()
        new_ids = ['GA', 'GT', 'badfeature', 'TT', 'AC', 'AA', 'AG', 'TA', 'AT', 'TG', 'GG', 'TC']
        self.assertListEqual(exp.feature_metadata.index.tolist(), new_ids)
        self.assertEqual(exp.shape, self.test1.shape)

    def test_sort_abundance_subgroup(self):
        exp = self.test1.sort_abundance(subgroup={'id': ['2']}, key=np.mean)
        new_ids = ['AC', 'TT', 'GA', 'GT', 'badfeature', 'AG', 'AA', 'TA', 'AT', 'TG', 'GG', 'TC']
        self.assertListEqual(exp.feature_metadata.index.tolist(), new_ids)
        self.assertEqual(exp.shape, self.test1.shape)

    def test_sort_ids_raise(self):
        fids = ['GG', 'pita']
        with self.assertRaisesRegex(ValueError, 'pita'):
            self.test1.sort_ids(fids)

    def test_sort_ids(self):
        # keep only samples S6 and S5
        new = self.test1.sort_ids(['S6', 'S5'], axis=0)

        # test sample_metadata are correct
        self.assertEqual(new.sample_metadata['id'][0], 6)
        self.assertEqual(new.sample_metadata['id'][1], 5)

        # test data are correct
        fid = 'GG'
        fpos = new.feature_metadata.index.get_loc(fid)
        self.assertEqual(new.data[0, fpos], 600)
        self.assertEqual(new.data[1, fpos], 500)


if __name__ == "__main__":
    main()
