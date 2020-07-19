# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

import numpy as np

from calour._testing import Tests
import calour as ca
from calour.ratio_experiment import RatioExperiment


class ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read_amplicon(self.rat1_biom, self.rat1_samp,
                                      min_reads=10, normalize=None)

    def test_from_exp(self):
        # default - no min reads threshold
        rexp = RatioExperiment.from_exp(self.test1, 'subj', 'time', '1', '2')
        # only 2 subjects have 2 time points
        self.assertEqual(rexp.shape[0], 2)
        # no features removed
        self.assertEqual(rexp.shape[1], self.test1.shape[1])
        # the 2 subjects are 1, 2
        self.assertListEqual(list(rexp.sample_metadata['subj']), [1, 2])
        self.assertEqual(rexp['S1', 'AA'], -1)
        self.assertEqual(rexp['S3', 'AG'], np.log2(600 / 100))
        self.assertTrue(np.isnan(rexp['S1', 'AC']))

        # supply threshold
        rexp = RatioExperiment.from_exp(self.test1, 'subj', 'time', '1', '2', threshold=400)
        # we should lose S1 since both are < threshold
        self.assertTrue(np.isnan(rexp['S1', 'AA']))
        # and D3 300 is corrected to 400
        self.assertEqual(rexp['S3', 'AA'], np.log2(400 / 500))

    def test_get_sign_pvals(self):
        rexp = RatioExperiment.from_exp(self.test1, 'subj', 'time', '1', '2')
        da_exp = rexp.get_sign_pvals(alpha=0.5, min_present=2)
        self.assertEqual(da_exp.shape[1], 3)
        self.assertEqual(da_exp.feature_metadata.loc['AA', '_calour_stat'], -1)
        self.assertEqual(da_exp.feature_metadata.loc['TA', '_calour_direction'], 'positive')

        da_exp = rexp.get_sign_pvals(alpha=1, min_present=1)
        self.assertEqual(da_exp.shape[1], 4)
        self.assertEqual(da_exp.feature_metadata.loc['AT', '_calour_pval'], 1)


if __name__ == "__main__":
    main()
