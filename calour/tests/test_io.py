# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import calour as ca
from calour._testing import Tests


class TestIO(Tests):

    def validate_read(self, exp, validate_sample_metadata=True):
        '''Validate the simple experiment was loaded correctly'''
        # number of bacteria is 12
        self.assertEqual(exp.data.shape[1], 12)
        # number of samples is 20 (should not read the samples only in map or only in biom table)
        # self.assertEqual(exp.data.shape[0],20)
        # test an OTU/sample to see it is in the right place
        sseq = ('TACGTAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGCGCA'
                'GGCGGTTTTGTAAGTCTGATGTGAAATCCCCGGGCTCAACCTGGGAATTG'
                'CATTGGAGACTGCAAGGCTAGAATCTGGCAGAGGGGGGTAGAATTCCACG')
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
        if validate_sample_metadata:
            self.assertEqual(exp.sample_metadata['id'][samplepos], 6)

    def test_read(self):
        # load the simple dataset as sparse
        exp = ca.read(self.simple_table, self.simple_map)
        self.validate_read(exp)

        # load the simple dataset as dense
        exp = ca.read(self.simple_table, self.simple_map, sparse=False)
        self.validate_read(exp)

        # test loading without a mapping file
        exp = ca.read(self.simple_table)
        self.validate_read(exp, validate_sample_metadata=False)


if __name__ == "__main__":
    unittest.main()
