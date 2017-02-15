# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
from tempfile import mkdtemp
from os.path import join
import shutil

import calour as ca
import numpy.testing as npt
import skbio

from calour._testing import Tests
from calour.io import _create_biom_table_from_exp


class TestIO(Tests):
    def setUp(self):
        super().setUp()
        # load the simple experiment as sparse
        self.simple = ca.read(self.simple_table, self.simple_map)
        self.outdir = mkdtemp()

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

    def test_create_biom_table_from_exp(self):
        exp = self.simple
        table = _create_biom_table_from_exp(exp)
        self.assertCountEqual(table.ids(axis='observation'), exp.feature_metadata.index.values)
        self.assertCountEqual(table.ids(axis='sample'), exp.sample_metadata.index.values)
        npt.assert_array_almost_equal(table.matrix_data.toarray(), exp.get_data(sparse=False).transpose())
        metadata = table.metadata(id=exp.feature_metadata.index[1], axis='observation')
        self.assertEqual(metadata['taxonomy'], exp.feature_metadata['taxonomy'].iloc[1])

    def test_save_fasta(self):
        exp = self.simple
        f = join(self.outdir, 'simple.fasta')
        exp.save_fasta(f)

        seqs = []
        for cseq in skbio.read(f, format='fasta'):
            seqs.append(str(cseq))
        self.assertCountEqual(seqs, exp.feature_metadata.index.values)
        shutil.rmtree(self.outdir)

if __name__ == "__main__":
    unittest.main()
