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
from numpy.testing import assert_almost_equal


class ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.pre_ratio = ca.read_amplicon(self.rat_pre_biom, self.rat_pre_samp,
                                          min_reads=10, normalize=None)
        self.ratio1 = ca.read(self.rat_biom, self.rat_samp, normalize=None, cls=RatioExperiment)

    def test_from_exp(self):
        # default - no min reads threshold
        rexp = RatioExperiment.from_exp(self.pre_ratio, 'subj', 'time', 1, 2)
        # only 2 subjects have 2 time points
        self.assertEqual(rexp.shape[0], 2)
        # no features removed
        self.assertEqual(rexp.shape[1], self.pre_ratio.shape[1])
        # the 2 subjects are 1, 2
        self.assertListEqual(list(rexp.sample_metadata['subj_1']), [1, 2])
        self.assertEqual(rexp['S1', 'AA'], 2)
        self.assertEqual(rexp['S3', 'AG'], np.log2(600 / 100))
        self.assertTrue(np.isnan(rexp['S1', 'AC']))
        # we double the amount of sample metadata fields (for nominator and denominator)
        self.assertEqual(len(rexp.sample_metadata.columns), 2 * len(self.pre_ratio.sample_metadata.columns))

        # supply threshold
        rexp = RatioExperiment.from_exp(self.pre_ratio, 'subj', 'time2', 1, 2, threshold=400)
        # we should lose S1 since both are < threshold
        self.assertTrue(np.isnan(rexp['S1', 'AA']))
        # and D3 300 is corrected to 400
        self.assertEqual(rexp['S3', 'AA'], np.log2(400 / 500))

        # supplying value2 as None
        rexp = RatioExperiment.from_exp(self.pre_ratio, 'subj', 'time2', 1, threshold=400)
        # Now we should not lose S1 since time2 value=3 is joined with value=2
        assert_almost_equal(rexp['S1', 'AA'], np.log2(400/((1500+400)/2)), decimal=3)
        # and D3 300 is corrected to 400
        self.assertEqual(rexp['S3', 'AA'], np.log2(400 / ((500+600)/2)))

    def test_get_sign_pvals(self):
        # rexp = RatioExperiment.from_exp(self.test1, 'subj', 'time', '1', '2')
        da_exp = self.ratio1.get_sign_pvals(alpha=0.5, min_present=2)
        self.assertEqual(da_exp.shape[1], 3)
        self.assertEqual(da_exp.feature_metadata.loc['AA', '_calour_stat'], -1)
        self.assertEqual(da_exp.feature_metadata.loc['TA', '_calour_direction'], 'positive')

        da_exp = self.ratio1.get_sign_pvals(alpha=1, min_present=1)
        self.assertEqual(da_exp.shape[1], 4)
        self.assertEqual(da_exp.feature_metadata.loc['AT', '_calour_pval'], 1)


if __name__ == "__main__":
    main()
