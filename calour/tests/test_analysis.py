# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import numpy as np

import calour as ca
from calour.analysis import diff_abundance
from calour._testing import Tests


class TestAnalysis(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        # load the complex experiment as sparse with normalizing and removing low read samples
        self.complex = ca.read_amplicon(self.timeseries_biom, self.timeseries_samp,
                                        min_reads=1000, normalize=10000)

    def test_diff_abundance(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test using defulat values
        dd = diff_abundance(self.test1, 'group', val1='1', val2='2')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        print(self.test1.feature_metadata.index.values)
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using no group 2 using only group 1
        dd = diff_abundance(self.test1, 'group', val1='1')
        expected_ids = [0, 1, 2, 3, 4, 10]
        # we get 1 less since now we also include badsample sample (not in the mapping file, so gets na)
        self.assertEqual(len(dd.feature_metadata), 6)
        for cid in expected_ids:
            print(self.test1.feature_metadata.index[cid])
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)
        # test using no val 2 using only group 2
        dd = diff_abundance(self.test1, 'group', val1='2')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            print(self.test1.feature_metadata.index[cid])
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_diff_abundance_alpha0(self):
        # Test when we should get 0 features (setting FDR level to 0)
        dd = diff_abundance(self.test1, 'group', val1='1', val2='2', alpha=0)
        self.assertEqual(dd.shape, (self.test1.shape[0], 0))

    def test_correlation_default(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test using spearman correlation
        dd = self.test1.correlation('id')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_nonzero_spearman(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test using non zero spearman correlation
        dd = self.test1.correlation('id', method='spearman', nonzero=True)
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_pearson(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test using pearson correlation
        dd = self.test1.correlation('id', method='pearson')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_nonzero_pearson(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test using non zero pearson correlation
        dd = self.test1.correlation('id', method='pearson', nonzero=True)
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)

    def test_correlation_complex(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test on real complex dataset (timeseries)
        dd = self.complex.correlation('MF_SAMPLE_NUMBER', method='pearson')
        self.assertTrue(np.abs(101 - len(dd.feature_metadata)) < 5)
        goodseq = 'TACGGAGGATGCGAGCGTTATTCGGAATCATTGGGTTTAAAGGGTCTGTAGGCGGGCTATTAAGTCAGGGGTGAAAGGTTTCAGCTTAACTGAGAAATTGCCTTTGATACTGGTAGTCTTGAATATCTGTGAAGTTCTTGGAATGTGTAG'
        self.assertIn(goodseq, dd.feature_metadata.index)
        goodseq = 'TACGTAGGTGGCAAGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGCAGGCGGATCAGTCAGTCTGTCTTAAAAGTTCGGGGCTTAACCCCGTGATGGGATGGAAACTGCTGATCTAGAGTATCGGAGAGGAAAGTGGAATTCCTAGT'
        self.assertIn(goodseq, dd.feature_metadata.index)

    def test_correlation_complex_spearman(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test on real complex dataset (timeseries) with spearman correlation
        dd = self.complex.correlation('MF_SAMPLE_NUMBER', method='spearman')
        self.assertTrue(np.abs(51 - len(dd.feature_metadata)) < 5)

    def test_diff_abundance_kw(self):
        np.random.seed(2017)
        dd = self.test1.diff_abundance_kw(field='group')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
        self.assertEqual(len(dd.feature_metadata), 7)
        for cid in expected_ids:
            self.assertIn(self.test1.feature_metadata.index[cid], dd.feature_metadata.index)


if __name__ == "__main__":
    unittest.main()
