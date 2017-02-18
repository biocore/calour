# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from copy import deepcopy

import calour as ca

from calour._testing import Tests, assert_experiment_equal


class IOTests(Tests):
    def setUp(self):
        super().setUp()
        self.simple = ca.read(self.test1_biom, self.test1_samp)
        self.simple2 = ca.read(self.test2_biom, self.test2_samp)

    def test_join_fields(self):
        # test the default params
        newexp = self.simple.join_fields('id', 'group', inplace=False)
        self.assertIn('id_group', newexp.sample_metadata.columns)
        self.assertEqual(newexp.sample_metadata.loc['S12', 'id_group'], '12.0_2')
        # test we didn't change anything besides the new sample metadata column
        assert_experiment_equal(newexp, self.simple, ignore_md_fields=['id_group'])

    def test_join_experiments(self):
        # do the famous join experiment to itself trick
        texp = deepcopy(self.simple)
        texp.description = 't2'
        newexp = self.simple.join_experiments(texp, prefixes=('c1', ''))
        self.assertEqual(len(newexp.feature_metadata), len(self.simple.feature_metadata))
        self.assertEqual(len(newexp.sample_metadata), len(self.simple.sample_metadata)*2)
        fexp = newexp.filter_samples('orig_exp', 't2')
        assert_experiment_equal(fexp, texp, ignore_md_fields=['orig_exp'])

if __name__ == "__main__":
    main()
