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


class TestSorting(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.simple = ca.read(self.simple_table, self.simple_map)
        # load the complex experiment as sparse
        self.complex = ca.read(self.complex_table, self.complex_map)

    def test_sort_taxonomy(self):
        newexp = self.simple.sort_taxonomy()
        expected_taxonomy = pd.Series.from_csv(join(self.test_data_dir, 'test1.sorted.taxonomy.csv'))
        pdt.assert_series_equal(newexp.feature_metadata['taxonomy'], expected_taxonomy, check_names=False)

    def test_cluster_features(self):
        # no minimal filtering
        # simple experiment
        newexp = self.simple.cluster_features()
        expected_experiment = ca.read(join(self.test_data_dir, 'test1.clustered.features.biom'), join(self.test_data_dir, 'test1.map.txt'))
        assert_experiment_equal(newexp, expected_experiment, check_history=False, almost_equal=True)
        # complex experiment (timeseries)
        newexp = self.complex.cluster_features()
        expected_experiment = ca.read(join(self.test_data_dir, 'timeseries.clustered.features.biom'), join(self.test_data_dir, 'timeseries.map.txt'))
        assert_experiment_equal(newexp, expected_experiment, check_history=False, almost_equal=True)


if __name__ == "__main__":
    unittest.main()
