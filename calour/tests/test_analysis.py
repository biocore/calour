# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import numpy as np
import numpy.testing as npt

import calour as ca
from calour.analysis import diff_abundance, get_term_features, relative_enrichment
from calour._testing import Tests


class TestAnalysis(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        # load the complex experiment as sparse with normalizing and removing low read samples
        self.complex = ca.read_amplicon(self.timeseries_biom, self.timeseries_samp,
                                        filter_reads=1000, normalize=10000)

    def test_diff_abundance(self):
        # set the seed as we are testing random permutations
        np.random.seed(2017)
        # test using defulat values
        dd = diff_abundance(self.test1, 'group', val1='1', val2='2')
        expected_ids = [0, 1, 2, 3, 4, 7, 10]
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

    def test_get_term_features(self):
        features = ['1', '2', '3', '4']
        feature_annotations = {'1': ['a', 'b'],
                               '2': ['a'],
                               '3': ['a', 'a'],
                               '5': ['c']}
        term_features = get_term_features(features, feature_annotations)
        self.assertCountEqual(list(term_features.keys()), ['a', 'b', 'c'])
        npt.assert_array_equal(term_features['a'], np.array([1, 1, 2, 0]))
        npt.assert_array_equal(term_features['b'], np.array([1, 0, 0, 0]))
        npt.assert_array_equal(term_features['c'], np.array([0, 0, 0, 0]))

    def test_relative_enrichment(self):
        f = self.test1.feature_metadata.index.values
        feature_terms = {}
        feature_terms[f[0]] = ['a', 'b', 'a', 'e', 'c', 'a', 'a', 'a', 'a', 'c', 'e']
        feature_terms[f[1]] = ['a', 'd', 'e', 'e']
        feature_terms[f[2]] = ['e', 'c', 'a', 'a', 'c', 'c', 'e']
        feature_terms[f[3]] = ['a', 'e', 'e', 'e']
        feature_terms[f[4]] = ['a', 'a', 'a', 'c', 'c', 'a', 'a', 'e', 'e']
        feature_terms[f[5]] = ['b', 'c', 'c', 'd', 'g', 'k']
        feature_terms[f[6]] = ['b', 'c', 'd', 'c', 'c', 'g', 'k']
        feature_terms[f[7]] = ['c', 'c', 'c', 'c', 'c']
        feature_terms[f[8]] = ['b', 'b', 'c', 'f', 'l', 'm', 'n']
        feature_terms[f[9]] = ['f']
        feature_terms[f[10]] = ['g', 'g', 'g', 'g', 'g', 'a']
        res = relative_enrichment(self.test1, f[:5], feature_terms)
        # we get 2 enriched features
        self.assertEqual(len(res), 2)
        # and the correct enrichment terms
        expected = [{'group2': 0.0, 'group1': 0.31428571428571428, 'description': 'e', 'observed': 11.0,
                     'pval': 0.0077273098637136162, 'expected': 5.7462686567164178},
                    {'group2': 0.03125, 'group1': 0.42857142857142855, 'description': 'a', 'observed': 15.0,
                     'pval': 0.0038208142131888057, 'expected': 8.3582089552238799}]
        self.assertCountEqual(res, expected)


if __name__ == "__main__":
    unittest.main()
