# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from copy import deepcopy

import pandas.testing as pdt
import numpy as np
import numpy.testing as npt

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


if __name__ == "__main__":
    main()
