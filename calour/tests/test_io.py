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
from numpy.testing import assert_array_almost_equal
import numpy as np
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
        exp = ca.read(self.test1_biom, self.test1_samp, self.test1_feat, normalize=None)
        self.assertTrue(scipy.sparse.issparse(exp.data))
        self._validate_read(exp)

    def test_read_openms_bucket_table(self):
        # load the openms bucket table with no metadata
        exp = ca.read(self.openms_csv, data_file_type='openms', sparse=False, normalize=None)
        self.assertEqual(len(exp.sample_metadata), 9)
        self.assertEqual(len(exp.feature_metadata), 10)
        self.assertEqual(exp.shape, (9, 10))
        self.assertEqual(exp.data[0, :].sum(), 8554202)
        self.assertEqual(exp.data[:, 1].sum(), 13795540)
        self.assertEqual(exp.sparse, False)

    def test_read_open_ms(self):
        exp = ca.read_open_ms(self.openms_csv, normalize=None)
        # test we get the MZ and RT correct
        self.assertIn('MZ', exp.feature_metadata)
        self.assertIn('RT', exp.feature_metadata)
        self.assertAlmostEqual(exp.feature_metadata['MZ'].iloc[1], 118.0869)
        self.assertAlmostEqual(exp.feature_metadata['RT'].iloc[1], 23.9214)
        # test normalizing
        exp = ca.read_open_ms(self.openms_csv, normalize=10000)
        assert_array_almost_equal(exp.data.sum(axis=1), np.ones(exp.shape[0])*10000)
        # test load sparse
        exp = ca.read_open_ms(self.openms_csv, sparse=True, normalize=None)
        self.assertEqual(exp.sparse, True)

    def test_read_qiim2(self):
        exp = ca.read(self.qiime2table, data_file_type='qiime2', normalize=None)
        self.assertEqual(exp.shape, (104, 658))

    def test_read_not_sparse(self):
        # load the simple dataset as dense
        exp = ca.read(self.test1_biom, self.test1_samp, sparse=False, normalize=None)
        self.assertFalse(scipy.sparse.issparse(exp.data))
        self._validate_read(exp)

    def test_read_no_sample_metadata(self):
        # test loading without a mapping file
        exp = ca.read(self.test1_biom, normalize=None)
        self._validate_read(exp, validate_sample_metadata=False)

    def test_read_amplicon(self):
        # test loading a taxonomy biom table and filtering/normalizing
        exp = ca.read_amplicon(self.test1_biom, filter_reads=1000, normalize=10000)
        exp2 = ca.read(self.test1_biom, normalize=None)
        exp2.filter_by_data('sum_abundance', cutoff=1000, inplace=True)
        exp2.normalize(inplace=True)
        assert_experiment_equal(exp, exp2)
        self.assertIn('taxonomy', exp.feature_metadata)

    def test_create_biom_table_from_exp(self):
        exp = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        table = _create_biom_table_from_exp(exp)
        self.assertCountEqual(table.ids(axis='observation'), exp.feature_metadata.index.values)
        self.assertCountEqual(table.ids(axis='sample'), exp.sample_metadata.index.values)
        assert_array_almost_equal(table.matrix_data.toarray(), exp.get_data(sparse=False).transpose())
        metadata = table.metadata(id=exp.feature_metadata.index[1], axis='observation')
        self.assertEqual(metadata['taxonomy'], exp.feature_metadata['taxonomy'].iloc[1])

    def test_save_fasta(self):
        exp = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        d = mkdtemp()
        f = join(d, 'test1.fasta')
        exp.save_fasta(f)
        seqs = []
        for seq in skbio.read(f, format='fasta'):
            seqs.append(str(seq))
        self.assertCountEqual(seqs, exp.feature_metadata.index.values)
        shutil.rmtree(d)

    def test_save_biom(self):
        # NOTE: Currently not testing the save biom hdf with taxonomy
        # as there is a bug there!
        exp = ca.read(self.test1_biom, self.test1_samp, normalize=None)
        d = mkdtemp()
        f = join(d, 'test1.save.biom')
        # test the json biom format
        exp.save_biom(f, fmt='json')
        newexp = ca.read(f, self.test1_samp, normalize=None)
        assert_experiment_equal(newexp, exp)
        # test the txt biom format
        exp.save_biom(f, fmt='txt')
        newexp = ca.read(f, self.test1_samp, normalize=None)
        assert_experiment_equal(newexp, exp, ignore_md_fields=['taxonomy'])
        # test the hdf5 biom format with no taxonomy
        exp.save_biom(f, add_metadata=None)
        newexp = ca.read(f, self.test1_samp, normalize=None)
        self.assertTrue('taxonomy' not in newexp.feature_metadata)
        assert_experiment_equal(newexp, exp, ignore_md_fields=['taxonomy'])
        shutil.rmtree(d)

    def test_save(self):
        exp = ca.read(self.test2_biom, self.test2_samp, normalize=None)
        d = mkdtemp()
        f = join(d, 'test1.save')
        # test the json biom format
        exp.save(f, fmt='json')
        newexp = ca.read(f+'.biom', f+'_sample.txt', normalize=None)
        assert_experiment_equal(newexp, exp, ignore_md_fields=['#SampleID.1'])
        shutil.rmtree(d)


if __name__ == "__main__":
    main()
