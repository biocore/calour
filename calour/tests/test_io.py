# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
from os.path import join, dirname, abspath

import calour as ca


class TestIO(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = join(dirname(abspath(__file__)), 'data')
        self.test_simple_table = join(self.test_data_dir, 'test1.biom.txt')
        self.test_simple_map = join(self.test_data_dir, 'test1.map.txt')
        self.test_complex_table = join(self.test_data_dir, 'timeseries.biom')
        self.test_complex_map = join(self.test_data_dir, 'timeseries.map.txt')

    def test_read(self):
        # load the simple dataset as sparse
        exp = ca.read(self.test_simple_table, self.test_simple_map)
        # number of bacteria is 12
        self.assertEqual(exp.data.shape[1],12)
        # number of samples is 20 (should not read the samples only in map or only in biom table)
        # self.assertEqual(exp.data.shape[0],20)
        # test an OTU/sample to see it is in the right place
        sseq = 'TACGTAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGCGCAGGCGGTTTTGTAAGTCTGATGTGAAATCCCCGGGCTCAACCTGGGAATTGCATTGGAGACTGCAAGGCTAGAATCTGGCAGAGGGGGGTAGAATTCCACG'
        ssample = 'S6'
        # test sample and sequence are in the table
        self.assertIn(sseq, exp.feature_metadata.index)
        self.assertIn(ssample, exp.sample_metadata.index)
        # test the location in the sample/feature metadata corresponds to the data
        samplepos = exp.sample_metadata.index.get_loc(ssample)
        seqpos = exp.feature_metadata.index.get_loc(sseq)
        self.assertEqual(exp.data[samplepos, seqpos], 6)
        # test the taxonomy is loaded correctly
        self.assertIn('g__Janthinobacterium', exp.feature_metadata['taxonomy'][seqpos])
        # test the sample metadata is loaded correctly
        self.assertEqual(exp.sample_metadata['id'][samplepos],6)

        # load the simple dataset as dense
        exp = ca.read(self.test_simple_table, self.test_simple_map, sparse=False)
        # number of bacteria is 12
        self.assertEqual(exp.data.shape[1],12)
        # number of samples is 20 (should not read the samples only in map or only in biom table)
        # self.assertEqual(exp.data.shape[0],20)
        # test an OTU/sample to see it is in the right place
        sseq = 'TACGTAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGCGCAGGCGGTTTTGTAAGTCTGATGTGAAATCCCCGGGCTCAACCTGGGAATTGCATTGGAGACTGCAAGGCTAGAATCTGGCAGAGGGGGGTAGAATTCCACG'
        ssample = 'S6'
        # test sample and sequence are in the table
        self.assertIn(sseq, exp.feature_metadata.index)
        self.assertIn(ssample, exp.sample_metadata.index)
        # test the location in the sample/feature metadata corresponds to the data
        samplepos = exp.sample_metadata.index.get_loc(ssample)
        seqpos = exp.feature_metadata.index.get_loc(sseq)
        self.assertEqual(exp.data[samplepos, seqpos], 6)
        # test the taxonomy is loaded correctly
        self.assertIn('g__Janthinobacterium',exp.feature_metadata['taxonomy'][seqpos])
        # test the sample metadata is loaded correctly
        self.assertEqual(exp.sample_metadata['id'][samplepos],6)

        # test loading without a mapping file
        # load the simple dataset as dense
        exp = ca.read(self.test_simple_table)
        # number of bacteria is 12
        self.assertEqual(exp.data.shape[1],12)
        # number of samples is 20 (should not read the samples only in map or only in biom table)
        # self.assertEqual(exp.data.shape[0],20)
        # test an OTU/sample to see it is in the right place
        sseq = 'TACGTAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGCGCAGGCGGTTTTGTAAGTCTGATGTGAAATCCCCGGGCTCAACCTGGGAATTGCATTGGAGACTGCAAGGCTAGAATCTGGCAGAGGGGGGTAGAATTCCACG'
        ssample = 'S6'
        # test sample and sequence are in the table
        self.assertIn(sseq, exp.feature_metadata.index)
        self.assertIn(ssample, exp.sample_metadata.index)
        # test the location in the sample/feature metadata corresponds to the data
        samplepos = exp.sample_metadata.index.get_loc(ssample)
        seqpos = exp.feature_metadata.index.get_loc(sseq)
        self.assertEqual(exp.data[samplepos, seqpos], 6)
        # test the taxonomy is loaded correctly
        self.assertIn('g__Janthinobacterium',exp.feature_metadata['taxonomy'][seqpos])

if __name__ == "__main__":
    unittest.main()
