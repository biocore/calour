# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import numpy.testing as npt
import numpy as np

import calour as ca
from calour.analysis import diff_abundance
from calour._testing import Tests


class TestAnalysis(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.simple = ca.read(self.simple_table, self.simple_map)
        # load the complex experiment as sparse
        self.complex = ca.read_taxa(self.complex_table, self.complex_map)

    def test_diff_abundance(self):
        # test using defulat values
        dd = diff_abundance(self.simple, 'group', val1='1', val2='2')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.simple.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using no group 2 using only group 1
        dd = diff_abundance(self.simple, 'group', val1='1')
        expected_ids = [0, 1, 2, 3, 4, 10]
        # we get 1 less since now we also include badsample sample (not in the mapping file, so gets na)
        self.assertEqual(len(dd.feature_metadata), 6)
        for cid in expected_ids:
            print(self.simple.feature_metadata.index[cid])
            self.assertIn(self.simple.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using no val 2 using only group 2
        dd = diff_abundance(self.simple, 'group', val1='2')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            print(self.simple.feature_metadata.index[cid])
            self.assertIn(self.simple.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using spearman correlation
        dd = diff_abundance(self.simple, 'id', method='spearman')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.simple.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using non zero spearman correlation
        dd = diff_abundance(self.simple, 'id', method='nonzerospearman')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.simple.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using pearson correlation
        dd = diff_abundance(self.simple, 'id', method='pearson')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.simple.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using non zero pearson correlation
        dd = diff_abundance(self.simple, 'id', method='nonzeropearson')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.simple.feature_metadata.index[cid], dd.feature_metadata.index)

        # test on real complex dataset (timeseries)
        dd = diff_abundance(self.complex, 'MF_SAMPLE_NUMBER', method='pearson')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertGreaterEqual(20, np.abs(len(dd.feature_metadata) - 40))
        goodseq = 'TACGGAGGATGCGAGCGTTATTCGGAATCATTGGGTTTAAAGGGTCTGTAGGCGGGCTATTAAGTCAGGGGTGAAAGGTTTCAGCTTAACTGAGAAATTGCCTTTGATACTGGTAGTCTTGAATATCTGTGAAGTTCTTGGAATGTGTAG'
        self.assertIn(goodseq, dd.feature_metadata.index)
        goodseq = 'TACGTAGGTGGCAAGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGCAGGCGGATCAGTCAGTCTGTCTTAAAAGTTCGGGGCTTAACCCCGTGATGGGATGGAAACTGCTGATCTAGAGTATCGGAGAGGAAAGTGGAATTCCTAGT'
        self.assertIn(goodseq, dd.feature_metadata.index)

if __name__ == "__main__":
    unittest.main()
