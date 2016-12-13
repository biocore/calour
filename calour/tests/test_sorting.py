# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
from os.path import join

import pandas as pd
import pandas.util.testing as pdt

import calour as ca
from calour._testing import Tests, assert_experiment_equal


class TestSorting(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.simple = ca.read(self.simple_table, self.simple_map)
        # load the complex experiment as sparse
        self.complex = ca.read(self.complex_table, self.complex_map)

    def test_sort_taxonomy(self):
        obs = self.simple.sort_taxonomy()
        expected_taxonomy = pd.Series.from_csv(join(self.test_data_dir, 'test1.sorted.taxonomy.csv'))
        pdt.assert_series_equal(obs.feature_metadata['taxonomy'], expected_taxonomy, check_names=False)

    def test_cluster_data(self):
        # no minimal filtering
        # simple experiment
        obs = self.simple.cluster_data()
        exp = ca.read(join(self.test_data_dir, 'test1.clustered.features.biom'), join(self.test_data_dir, 'test1.map.txt'))
        assert_experiment_equal(obs, exp, check_history=False, almost_equal=True)
        # complex experiment (timeseries)
        obs = self.complex.cluster_data()
        exp = ca.read(join(self.test_data_dir, 'timeseries.clustered.features.biom'), join(self.test_data_dir, 'timeseries.map.txt'))
        assert_experiment_equal(obs, exp, check_history=False, almost_equal=True)

    def test_sort_by_metadata(self):
        # test sorting inplace and various fields (keeping the order)
        obs = self.complex.sort_by_metadata(field='MINUTES')
        obs = obs.sort_by_metadata(field='HOUR')
        obs.sort_by_metadata(field='DAY', inplace=True)
        exp = ca.read(join(self.test_data_dir, 'timeseries.sorted.time.biom'), join(self.test_data_dir, 'timeseries.map.txt'))
        assert_experiment_equal(obs, exp, check_history=False, almost_equal=True)
        # also test first and last samples are ok
        self.assertEqual(obs.sample_metadata['MF_SAMPLE_NUMBER'][0], 1)
        self.assertEqual(obs.sample_metadata['MF_SAMPLE_NUMBER'][-1], 96)

    def test_sort_by_data(self):
        obs = self.complex.sort_by_data(axis=1)
        exp = ca.read(join(self.test_data_dir, 'timeseries.sorted.freq.biom'), join(self.test_data_dir, 'timeseries.map.txt'))
        assert_experiment_equal(obs, exp, check_history=False, almost_equal=True)


if __name__ == "__main__":
    unittest.main()
