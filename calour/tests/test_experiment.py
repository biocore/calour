# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
from os.path import join, dirname, abspath

import numpy as np
import numpy.testing as npt

from calour._testing import Tests
import calour as ca


class TestExperiment(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.simple = ca.read(self.simple_table, self.simple_map)
        # load the simple experiment as dense
        self.simple_dense = ca.read(self.simple_table, self.simple_map, sparse=False)
        # load the complex experiment as sparse
        self.complex = ca.read(self.complex_table, self.complex_map)

    def test_reorder(self):
        exp = self.simple
        # keep only samples 5,4
        newexp = exp.reorder([5, 4], axis=0)

        # test we didn't loose any bacteria, correct number of samples
        self.assertEqual(newexp.data.shape[0], 2)
        self.assertEqual(newexp.data.shape[1], exp.data.shape[1])

        # test both sample_metadata and data are correct
        self.assertEqual(newexp.sample_metadata['id'][0], 6)
        self.assertEqual(newexp.sample_metadata['id'][1], 5)
        sseq = 'TACGTAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGCGCAGGCGGTTTTGTAAGTCTGATGTGAAATCCCCGGGCTCAACCTGGGAATTGCATTGGAGACTGCAAGGCTAGAATCTGGCAGAGGGGGGTAGAATTCCACG'
        seqpos = exp.feature_metadata.index.get_loc(sseq)
        self.assertEqual(newexp.data[0, seqpos], 6)
        self.assertEqual(newexp.data[1, seqpos], 5)

        # test inplace reordering of features
        sseq = 'TACGTAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGTGCGCAGGCGGTTTTGTAAGTCTGATGTGAAATCCCCGGGCTCAACCTGGGAATTGCATTGGAGACTGCAAGGCTAGAATCTGGCAGAGGGGGGTAGAATTCCACG'
        seqpos = exp.feature_metadata.index.get_loc(sseq)
        newexp.reorder([seqpos, 0], axis=1, inplace=True)
        seqpos = newexp.feature_metadata.index.get_loc(sseq)
        self.assertEqual(seqpos, 0)
        self.assertEqual(newexp.data[0, 0], 6)
        self.assertEqual(newexp.data[1, 0], 5)

        # test double permuting of big dataset
        exp = self.complex
        rand_perm_samples = np.random.permutation(exp.data.shape[0])
        rand_perm_features = np.random.permutation(exp.data.shape[1])
        rev_perm_samples = np.argsort(rand_perm_samples)
        rev_perm_features = np.argsort(rand_perm_features)
        newexp = exp.reorder(rand_perm_features, axis=1)
        newexp = newexp.reorder(rand_perm_samples, axis=0, inplace=True)
        newexp = newexp.reorder(rev_perm_features, axis=1, inplace=True)
        newexp = newexp.reorder(rev_perm_samples, axis=0)
        self.assertEqual(0, np.sum(newexp.data != exp.data))
        self.assertTrue(newexp.sample_metadata.equals(exp.sample_metadata))
        self.assertTrue(newexp.feature_metadata.equals(exp.feature_metadata))

if __name__ == "__main__":
    unittest.main()
