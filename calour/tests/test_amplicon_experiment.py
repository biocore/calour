# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from copy import deepcopy
from os.path import join

import pandas.util.testing as pdt
import numpy as np
import numpy.testing as npt
import pandas as pd

from calour._testing import Tests
import calour as ca


class ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read_amplicon(self.test1_biom, self.test1_samp, normalize=True)

    def test_filter_taxonomy(self):
        # default - substring and keep matching
        exp = self.test1.filter_taxonomy('proteobacteria')
        self.assertEqual(exp.shape[1], 2)
        self.assertEqual(set(exp.feature_metadata.index), set(self.test1.feature_metadata.index[[2, 3]]))
        # check we didn't change the samples
        pdt.assert_frame_equal(exp.sample_metadata, self.test1.sample_metadata)

        # test with list of values and negate
        exp = self.test1.filter_taxonomy(['Firmicutes', 'proteobacteria'], negate=True)
        self.assertEqual(exp.shape[1], 6)
        # should not have this sequence
        self.assertNotIn(self.test1.feature_metadata.index[4], exp.feature_metadata.index)
        # should have all these sequences
        okseqs = ['TACGTATGTCACAAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCCGTGGATTAAGCGTGTTGTGAAATGTAGACGCTCAACGTCTGAATCGCAGCGCGAACTGGTTCACTTGAGTATGCACAACGTAGGCGGAATTCGTCG',
                  'TACATAGGTCGCAAGCGTTATCCGGAATTATTGGGCGTAAAGCGTTCGTAGGCTGTTTATTAAGTCTGGAGTCAAATCCCAGGGCTCAACCCTGGCTCGCTTTGGATACTGGTAAACTAGAGTTAGATAGAGGTAAGCAGAATTCCATGT',
                  'TACGGAGGATGCGAGCGTTATCTGGAATCATTGGGTTTAAAGGGTCCGTAGGCGGGTTGATAAGTCAGAGGTGAAAGCGCTTAGCTCAACTAAGCAACTGCCTTTGAAACTGTCAGTCTTGAATGATTGTGAAGTAGTTGGAATGTGTAG',
                  'TACGTAGGGCGCGAGCGTTGTCCGGAATTATTGGGCGTAAAGGGCTTGTAGGCGGTTGGTCGCGTCTGCCGTGAAATTCTCTGGCTTAACTGGAGGCGTGCGGTGGGTACGGGCTGACTTGAGTGCGGTAGGGGAGACTGGAACTCCTGG',
                  'AAAAAAAGGTCCAGGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCGGACGATTAAGTCAGCTGCGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGAAACTGGTTGTCTTGAGTGCACGCAGGGATGTTGGAATTCATGG',
                  'ACGT']
        for cseq in okseqs:
            self.assertIn(cseq, exp.feature_metadata.index)

    def test_filter_fasta(self):
        # test keeping the sequences from fasta
        exp = self.test1.filter_fasta(self.seqs1_fasta)
        # test we get only 1 sequence and the correct one
        self.assertEqual(len(exp.feature_metadata), 1)
        self.assertEqual(exp.shape[1], 1)
        self.assertEqual(exp.feature_metadata.index[0], self.test1.feature_metadata.index[5])
        # and same number of samples
        self.assertEqual(exp.shape[0], self.test1.shape[0])
        # and data is ok
        data = exp.get_data(sparse=False)
        orig_data = self.test1.get_data(sparse=False)
        npt.assert_array_equal(data[:, 0], orig_data[:, 5])
        # and is not inplace
        self.assertIsNot(exp, self.test1)

    def test_filter_fasta_inverse(self):
        # test removing sequences from fasta and inplace
        orig_exp = deepcopy(self.test1)
        exp = self.test1.filter_fasta(self.seqs1_fasta, negate=True, inplace=True)
        # test we remove only 1 sequence and the correct one
        self.assertEqual(len(exp.feature_metadata), orig_exp.shape[1] - 1)
        self.assertEqual(exp.shape[1], orig_exp.shape[1] - 1)
        self.assertNotIn(orig_exp.feature_metadata.index[5], exp.feature_metadata.index)
        # and same number of samples
        self.assertEqual(exp.shape[0], orig_exp.shape[0])
        # and data is ok
        data = exp.get_data(sparse=False)
        orig_data = orig_exp.get_data(sparse=False)
        okseqs = np.hstack([np.arange(5), np.arange(6, 12)])
        npt.assert_array_equal(data, orig_data[:, okseqs])
        # and is inplace
        self.assertIs(exp, self.test1)

    def test_sort_taxonomy(self):
        obs = self.test1.sort_taxonomy()
        expected_taxonomy = pd.Series.from_csv(join(self.test_data_dir, 'test1.sorted.taxonomy.csv'))
        pdt.assert_series_equal(obs.feature_metadata['taxonomy'], expected_taxonomy, check_names=False)

    def test_filter_orig_reads(self):
        obs = self.test1.filter_orig_reads(2900)
        self.assertEqual(obs.shape[0], 2)
        self.assertIn('S19', obs.sample_metadata.index)
        self.assertIn('S20', obs.sample_metadata.index)
        self.assertEqual(obs.shape[1], self.test1.shape[1])


if __name__ == "__main__":
    main()
