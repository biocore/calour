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

import calour as ca

from calour._testing import Tests, assert_experiment_equal


class MTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)

    def test_join_fields(self):
        # test the default params
        newexp = self.test1.join_fields('id', 'group', inplace=False)
        self.assertIn('id_group', newexp.sample_metadata.columns)
        self.assertEqual(newexp.sample_metadata.loc['S12', 'id_group'], '12.0_2')
        # test we didn't change anything besides the new sample metadata column
        assert_experiment_equal(newexp, self.test1, ignore_md_fields=['id_group'])

    def test_join_fields_complex(self):
        # test join feature fields with new field name, separator and inplace
        exp = deepcopy(self.test1)
        newexp = exp.join_fields('taxonomy', 'taxonomy', newname='test', axis=1, sep=';', inplace=True)
        self.assertIs(newexp, exp)
        self.assertIn('test', exp.feature_metadata.columns)
        self.assertNotIn('test', exp.sample_metadata.columns)
        self.assertEqual(exp.feature_metadata['test'].iloc[11], 'bad_bacteria;bad_bacteria')
        # test we didn't change anything besides the new sample metadata column
        assert_experiment_equal(exp, self.test1, ignore_md_fields=['test'])

    def test_join_experiments(self):
        # do the famous join experiment to itself trick
        texp = deepcopy(self.test1)
        texp.description = 't2'
        newexp = self.test1.join_experiments(texp, prefixes=('c1', ''))
        self.assertEqual(len(newexp.feature_metadata), len(self.test1.feature_metadata))
        self.assertEqual(len(newexp.sample_metadata), len(self.test1.sample_metadata)*2)
        fexp = newexp.filter_samples('orig_exp', ['t2'])
        assert_experiment_equal(fexp, texp, ignore_md_fields=['orig_exp'])

    def test_merge_identical(self):
        # test default conditions - on samples, not inplace, mean method
        newexp = self.test1.merge_identical('group')
        self.assertEqual(newexp.shape[0], 3)
        self.assertEqual(list(newexp.data[:, 3]), [0, 10, 5])
        self.assertIsNot(newexp, self.test1)
        self.assertEqual(newexp.shape[1], self.test1.shape[1])
        # test the counts/original samples per merge value
        self.assertCountEqual(newexp.sample_metadata['_calour_merge_ids']['S1'],
                              ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11'])
        self.assertCountEqual(newexp.sample_metadata['_calour_merge_ids']['S12'],
                              ['S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20'])
        self.assertEqual(newexp.sample_metadata['_calour_merge_number']['S1'], 11)
        self.assertEqual(newexp.sample_metadata['_calour_merge_number']['S12'], 9)

    def test_merge_identical_sum(self):
        # test on samples, inplace, sum method
        newexp = self.test1.merge_identical('group', method='sum', inplace=True)
        newexp.sparse = False
        self.assertEqual(newexp.shape[0], 3)
        self.assertEqual(list(newexp.data[:, 3]), [0, 90, 5])
        self.assertIs(newexp, self.test1)
        self.assertEqual(newexp.shape[1], 12)

    def test_merge_identical_random_unique(self):
        # test on features, random method, not inplace
        # since each taxonomy is unique, should have the same as original
        newexp = self.test1.merge_identical('taxonomy', method='random', axis=1)
        self.assertEqual(newexp.shape, self.test1.shape)
        self.assertIsNot(newexp, self.test1)

    def test_merge_identical_random(self):
        # test on samples, random method, not inplace
        np.random.seed(2017)
        newexp = self.test1.merge_identical('group', method='random')
        self.assertEqual(newexp.shape[0], 3)
        self.assertEqual(list(newexp.data[:, 7]), [849, 859, 9])
        self.assertEqual(newexp.shape[1], self.test1.shape[1])
        self.assertIsNot(newexp, self.test1)
        np.random.seed(2018)
        newexp = self.test1.merge_identical('group', method='random')
        self.assertNotEqual(list(newexp.data[:, 7]), [849, 859, 9])


if __name__ == "__main__":
    main()
