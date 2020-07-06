# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from copy import deepcopy

import numpy as np
import pandas as pd

import calour as ca
from calour._testing import Tests


class MTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test1_biom, self.test1_samp, self.test1_feat, normalize=None)

    def test_join_experiments(self):
        # do the famous join experiment to itself trick
        texp = deepcopy(self.test1)
        texp.description = 't2'
        newexp = self.test1.join_experiments(texp, prefixes=('c1', ''))
        self.assertEqual(len(newexp.feature_metadata), len(self.test1.feature_metadata))
        self.assertEqual(len(newexp.sample_metadata), len(self.test1.sample_metadata)*2)
        fexp = newexp.filter_samples('experiments', ['t2'])
        self.assert_experiment_equal(fexp, texp, ignore_md_fields=['experiments'])

    def test_join_experiments_featurewise(self):
        otu1 = ca.Experiment(np.array([[0, 9], [7, 4]]), sparse=False,
                             sample_metadata=pd.DataFrame({'category': ['B', 'A'],
                                                           'ph': [7.7, 6.6]},
                                                          index=['s2', 's1']),
                             feature_metadata=pd.DataFrame({'motile': ['y', 'n']}, index=['16S1', '16S2']))
        otu2 = ca.Experiment(np.array([[6], [8], [10]]), sparse=False,
                             sample_metadata=pd.DataFrame({'category': ['A', 'B', 'C'],
                                                           'ph': [6.6, 7.7, 8.8]},
                                                          index=['s1', 's2', 's3']),
                             feature_metadata=pd.DataFrame({'motile': [None]}, index=['ITS1']))
        combined_obs = otu1.join_experiments_featurewise(otu2, 'origin', ('16S', 'ITS'))
        combined_exp = ca.Experiment(np.array([[7, 4, 6], [0, 9, 8]]), sparse=False,
                                     sample_metadata=pd.DataFrame({'category': ['A', 'B'],
                                                                   'ph': [6.6, 7.7]},
                                                                  index=['s1', 's2']),
                                     feature_metadata=pd.DataFrame({'motile': ['y', 'n', None],
                                                                    'origin': ['16S', '16S', 'ITS']},
                                                                   index=['16S1', '16S2', 'ITS1']))
        # reorder the samples
        combined_obs = combined_obs.filter_ids(combined_exp.sample_metadata.index, axis=0)
        self.assert_experiment_equal(combined_obs, combined_exp)

    def test_agg_by_metadata(self):
        # test default conditions - on samples, not inplace, mean method
        newexp = self.test1.aggregate_by_metadata('group')
        self.assertEqual(newexp.shape[0], 3)
        self.assertEqual(list(newexp.data[:, 3]), [0, 10, 5])
        self.assertIsNot(newexp, self.test1)
        self.assertEqual(newexp.shape[1], self.test1.shape[1])
        # test the counts/original samples per merge value
        self.assertCountEqual(newexp.sample_metadata['_calour_merge_ids']['S1'],
                              ';'.join(['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']))
        self.assertCountEqual(newexp.sample_metadata['_calour_merge_ids']['S12'],
                              ';'.join(['S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']))
        self.assertEqual(newexp.sample_metadata['_calour_merge_number']['S1'], 11)
        self.assertEqual(newexp.sample_metadata['_calour_merge_number']['S12'], 9)

    def test_agg_by_metadata_sum(self):
        # test on samples, inplace, sum method
        newexp = self.test1.aggregate_by_metadata('group', 'sum', inplace=True)
        newexp.sparse = False
        self.assertEqual(newexp.shape[0], 3)
        self.assertEqual(list(newexp.data[:, 3]), [0, 90, 5])
        self.assertIs(newexp, self.test1)
        self.assertEqual(newexp.shape[1], 12)


if __name__ == "__main__":
    main()
