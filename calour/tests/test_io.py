# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

import scipy.sparse

import calour as ca
from calour._testing import Tests


class IOTests(Tests):
    def _validate_read(self, exp, validate_sample_metadata=True):
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
        exp = ca.read(self.test1_biom, self.test1_samp, self.test1_feat)
        self.assertTrue(scipy.sparse.issparse(exp.data))
        self._validate_read(exp)

    def test_read_not_sparse(self):
        # load the simple dataset as dense
        exp = ca.read(self.test1_biom, self.test1_samp, sparse=False)
        self.assertFalse(scipy.sparse.issparse(exp.data))
        self._validate_read(exp)

    def test_read_no_sample_metadata(self):
        # test loading without a mapping file
        exp = ca.read(self.test1_biom)
        self._validate_read(exp, validate_sample_metadata=False)


if __name__ == "__main__":
    main()
