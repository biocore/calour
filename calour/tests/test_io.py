# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from tempfile import mkdtemp
from os.path import join
import shutil

import scipy.sparse
import numpy.testing as npt
import skbio

import calour as ca

from calour._testing import Tests, assert_experiment_equal

from calour.io import _create_biom_table_from_exp


class IOTests(Tests):
    def setUp(self):
        super().setUp()

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

    def test_read_openms_bucket_table(self):
        # load the openms bucket table with no metadata
        exp = ca.read(self.openms_csv, data_file_type='openms', sparse=False)
        self.assertEqual(len(exp.sample_metadata), 9)
        self.assertEqual(len(exp.feature_metadata), 10)
        self.assertEqual(exp.shape, (9, 10))
        self.assertEqual(exp.data[0, :].sum(), 8554202)
        self.assertEqual(exp.data[:, 1].sum(), 13795540)
        self.assertEqual(exp.sparse, False)

    def test_read_not_sparse(self):
        # load the simple dataset as dense
        exp = ca.read(self.test1_biom, self.test1_samp, sparse=False)
        self.assertFalse(scipy.sparse.issparse(exp.data))
        self._validate_read(exp)

    def test_read_no_sample_metadata(self):
        # test loading without a mapping file
        exp = ca.read(self.test1_biom)
        self._validate_read(exp, validate_sample_metadata=False)

    def test_read_taxa(self):
        # test loading a taxonomy biom table and filtering/normalizing
        exp = ca.read_taxa(self.test1_biom)
        exp2 = ca.read(self.test1_biom)
        exp2.filter_by_data('sum_abundance', cutoff=1000, inplace=True)
        exp2.normalize(inplace=True)
        assert_experiment_equal(exp, exp2)
        self.assertIn('taxonomy', exp.feature_metadata)

    def test_create_biom_table_from_exp(self):
        exp = ca.read(self.test1_biom, self.test1_samp)
        table = _create_biom_table_from_exp(exp)
        self.assertCountEqual(table.ids(axis='observation'), exp.feature_metadata.index.values)
        self.assertCountEqual(table.ids(axis='sample'), exp.sample_metadata.index.values)
        npt.assert_array_almost_equal(table.matrix_data.toarray(), exp.get_data(sparse=False).transpose())
        metadata = table.metadata(id=exp.feature_metadata.index[1], axis='observation')
        self.assertEqual(metadata['taxonomy'], exp.feature_metadata['taxonomy'].iloc[1])

    def test_save_fasta(self):
        exp = ca.read(self.test1_biom, self.test1_samp)
        d = mkdtemp()
        f = join(d, 'test1.fasta')
        exp.save_fasta(f)
        seqs = []
        for seq in skbio.read(f, format='fasta'):
            seqs.append(str(seq))
        self.assertCountEqual(seqs, exp.feature_metadata.index.values)
        shutil.rmtree(d)


if __name__ == "__main__":
    main()
