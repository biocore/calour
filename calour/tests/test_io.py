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
from io import StringIO
import shutil
import logging

import scipy.sparse
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

import calour as ca
from calour._testing import Tests, assert_experiment_equal
from calour.io import _create_biom_table_from_exp
from calour.util import _iter_fasta


class IOTests(Tests):
    def setUp(self):
        super().setUp()

    def _validate_read(self, exp, validate_sample_metadata=True):
        # number of bacteria is 12 in biom table
        self.assertEqual(exp.data.shape[1], 12)
        # number of samples is 21 (should not read the samples only in mapping file)
        self.assertEqual(exp.data.shape[0], 21)
        # test an OTU/sample to see it is in the right place
        fid = 'GG'
        sid = 'S12'
        # test sample and sequence are in the table
        self.assertIn(fid, exp.feature_metadata.index)
        self.assertIn(sid, exp.sample_metadata.index)
        # test the location in the sample/feature metadata corresponds to the data
        spos = exp.sample_metadata.index.get_loc(sid)
        fpos = exp.feature_metadata.index.get_loc(fid)
        # there is only one cell with value of 1200
        self.assertEqual(exp.data[spos, fpos], 1200)
        # test the taxonomy is loaded correctly
        self.assertEqual('Unknown', exp.feature_metadata['taxonomy'][fid])
        # test the sample metadata is loaded correctly
        if validate_sample_metadata:
            self.assertEqual(exp.sample_metadata['id'][spos], 12)

    def test_read_metadata(self):
        # test it's ok to read the IDs of numbers as str
        f = StringIO('''SampleID	foo
0100.02	a
100.030	b
''')
        try:
            ca.io._read_metadata(['0100.02', '100.030'], f, None)
        except:
            self.fail('Should not raise exception while reading metadata.')

    def test_read(self):
        # re-enable logging because it is disabled in setUp
        logging.disable(logging.NOTSET)
        with self.assertLogs(level='INFO') as cm:
            # load the simple dataset as sparse
            exp = ca.read(self.test1_biom, self.test1_samp, self.test1_feat, normalize=None)
            # test the log messages are correct
            self.assertRegex(cm.output[0], 'loaded 21 samples, 12 features')
            self.assertRegex(cm.output[1], "dropped \\(1\\): {'SAMPLE_NOT_EXIST'}")
            self.assertRegex(cm.output[2], "These have data but do not have metadata: {'badsample'}")
            self.assertRegex(cm.output[3], "dropped \\(1\\): {'FEATURE_NOT_EXIST'}")
            self.assertRegex(cm.output[4], "These have data but do not have metadata: {'badfeature'}")

            self.assertTrue(scipy.sparse.issparse(exp.data))
            self._validate_read(exp)

    def test_read_not_sparse(self):
        logging.disable(logging.NOTSET)
        with self.assertLogs(level='INFO') as cm:
            # load the simple dataset as dense
            exp = ca.read(self.test1_biom, self.test1_samp, sparse=False, normalize=None)
            self.assertFalse(scipy.sparse.issparse(exp.data))
            self._validate_read(exp, cm.output)

    def test_read_sample_kwargs(self):
        # re-enable logging because it is disabled in setUp
        logging.disable(logging.NOTSET)
        with self.assertLogs(level='INFO') as cm:
            # load the simple dataset as sparse
            exp = ca.read(self.test1_biom, self.test1_samp, self.test1_feat, normalize=None,
                          sample_metadata_kwargs={'parse_dates': ['collection_date']})
            # test the log messages are correct
            self.assertRegex(cm.output[0], 'loaded 21 samples, 12 features')
            self.assertRegex(cm.output[1], "dropped \\(1\\): {'SAMPLE_NOT_EXIST'}")
            self.assertRegex(cm.output[2], "These have data but do not have metadata: {'badsample'}")
            self.assertRegex(cm.output[3], "dropped \\(1\\): {'FEATURE_NOT_EXIST'}")
            self.assertRegex(cm.output[4], "These have data but do not have metadata: {'badfeature'}")

            self.assertTrue(scipy.sparse.issparse(exp.data))
            self._validate_read(exp)

            obs_dates = exp.sample_metadata['collection_date'].tolist()
            # the last sample in OTU table does not have metadata, so NaT
            exp_dates = [pd.Timestamp('2017-8-1')] * 20 + [pd.NaT]
            self.assertListEqual(obs_dates, exp_dates)

    def test_read_feature_kwargs(self):
        # re-enable logging because it is disabled in setUp
        logging.disable(logging.NOTSET)
        with self.assertLogs(level='INFO') as cm:
            # load the simple dataset as sparse
            exp = ca.read(self.test1_biom, self.test1_samp, self.test1_feat, normalize=None,
                          feature_metadata_kwargs={'dtype': {'ph': str}})
            # test the log messages are correct
            self.assertRegex(cm.output[0], 'loaded 21 samples, 12 features')
            self.assertRegex(cm.output[1], "dropped \\(1\\): {'SAMPLE_NOT_EXIST'}")
            self.assertRegex(cm.output[2], "These have data but do not have metadata: {'badsample'}")
            self.assertRegex(cm.output[3], "dropped \\(1\\): {'FEATURE_NOT_EXIST'}")
            self.assertRegex(cm.output[4], "These have data but do not have metadata: {'badfeature'}")

            self.assertTrue(scipy.sparse.issparse(exp.data))
            self._validate_read(exp)
            # read as str not float
            self.assertEqual(exp.feature_metadata.loc['AA', 'ph'], '4.0')

    def test_read_no_metadata(self):
        logging.disable(logging.NOTSET)
        with self.assertLogs(level='INFO') as cm:
            # test loading without a mapping file
            exp = ca.read(self.test1_biom, normalize=None)
            self.assertRegex(cm.output[0], 'loaded 21 samples, 12 features')
            self._validate_read(exp, validate_sample_metadata=False)

    def test_read_amplicon(self):
        # test loading a taxonomy biom table and filtering/normalizing
        exp1 = ca.read_amplicon(self.test1_biom, min_reads=1000, normalize=10000)
        exp2 = ca.read(self.test1_biom, normalize=None)
        exp2.filter_by_data('abundance', axis=0, cutoff=1000, inplace=True, mean_or_sum='sum')
        exp2.normalize(inplace=True)
        assert_experiment_equal(exp1, exp2)
        self.assertIn('taxonomy', exp1.feature_metadata.columns)

    def test_read_openms_bucket_table(self):
        # load the openms bucket table with no metadata
        exp = ca.read(self.openms_csv, data_file_type='csv', sparse=False, normalize=None)
        self.assertEqual(len(exp.sample_metadata), 9)
        self.assertEqual(len(exp.feature_metadata), 10)
        self.assertEqual(exp.shape, (9, 10))
        self.assertEqual(exp.data[0, :].sum(), 8554202)
        self.assertEqual(exp.data[:, 1].sum(), 13795540)
        self.assertEqual(exp.sparse, False)

    def test_read_openms_bucket_table_samples_are_rows(self):
        # load the openms bucket table with no metadata
        exp = ca.read(self.openms_samples_rows_csv, data_file_type='csv', sample_in_row=True, sparse=False, normalize=None)
        self.assertEqual(len(exp.sample_metadata), 9)
        self.assertEqual(len(exp.feature_metadata), 10)
        self.assertEqual(exp.shape, (9, 10))
        self.assertEqual(exp.data[0, :].sum(), 8554202)
        self.assertEqual(exp.data[:, 1].sum(), 13795540)
        self.assertEqual(exp.sparse, False)

    def test_read_open_ms(self):
        exp = ca.read_ms(self.openms_csv, normalize=None, data_file_type='openms')
        # test we get the MZ and RT correct
        self.assertIn('MZ', exp.feature_metadata)
        self.assertIn('RT', exp.feature_metadata)
        self.assertEqual(exp.feature_metadata['MZ'].iloc[1], 118.0869)
        self.assertEqual(exp.feature_metadata['RT'].iloc[1], 23.9214)
        # test normalizing
        exp = ca.read_ms(self.openms_csv, normalize=10000, data_file_type='openms')
        assert_array_almost_equal(exp.data.sum(axis=1), np.ones(exp.shape[0]) * 10000)
        # test load sparse
        exp = ca.read_ms(self.openms_csv, sparse=True, normalize=None, data_file_type='openms')
        self.assertEqual(exp.sparse, True)

    def test_read_biom_ms(self):
        # load a biom table with MZ/RT in featureID, and associated gnps clusterinfo file
        exp = ca.read_ms(self.ms_biom_table, sample_metadata_file=self.gnps_map,
                         data_file_type='biom', use_gnps_id_from_AllFiles=False, normalize=None)
        self.assertIn('MZ', exp.feature_metadata)
        self.assertIn('RT', exp.feature_metadata)
        self.assertEqual(exp.feature_metadata['MZ'].iloc[1], 899.53)
        self.assertEqual(exp.feature_metadata['RT'].iloc[0], 314)

    def test_read_mzmine2_ms(self):
        # load an mzmine2 metabolomics table, and associated gnps clusterinfo file
        exp = ca.read_ms(self.mzmine2_csv, sample_metadata_file=self.gnps_map,
                         data_file_type='mzmine2', use_gnps_id_from_AllFiles=False, normalize=None)
        self.assertIn('MZ', exp.feature_metadata)
        self.assertIn('RT', exp.feature_metadata)
        self.assertEqual(exp.feature_metadata['MZ'].iloc[1], 200)
        self.assertEqual(exp.feature_metadata['RT'].iloc[0], 1)
        self.assertEqual(exp.data[2, 1], 35900)

    def test_read_mzmine2_ms_with_idstr(self):
        # load an mzmine2 metabolomics table with the sampleids inflated with additional info
        exp = ca.read_ms(self.mzmine2_with_idstr_csv, sample_metadata_file=self.gnps_map,
                         use_gnps_id_from_AllFiles=False, cut_sample_id_sep='_', normalize=None)
        self.assertEqual(exp.feature_metadata['MZ'].iloc[1], 200)
        self.assertEqual(exp.feature_metadata['RT'].iloc[0], 1)
        self.assertEqual(exp.sample_metadata['field2'][0], 'f')
        self.assertEqual(exp.data[2, 1], 35900)
        self.assertEqual(exp.data.shape, (6, 6))

    def test_read_gnps_ms(self):
        # load the gnps exported table with associated sample metadata and cluster info
        exp = ca.read_ms(self.gnps_table, sample_metadata_file=self.gnps_map,
                         data_file_type='gnps-ms2', normalize=None)
        # verify the load extracts required fields to metadata
        self.assertEqual(exp.data[2, 3], 139692)
        # # test normalizing
        exp = ca.read_ms(self.gnps_table, sample_metadata_file=self.gnps_map,
                         data_file_type='gnps-ms2', normalize=10000)
        assert_array_almost_equal(exp.data.sum(axis=1), np.ones(exp.shape[0]) * 10000)
        # # test load sparse
        exp = ca.read_ms(self.gnps_table, sample_metadata_file=self.gnps_map,
                         data_file_type='gnps-ms2', normalize=None, sparse=True)
        self.assertEqual(exp.sparse, True)

    def test_read_open_ms_samples_rows(self):
        exp = ca.read_ms(self.openms_samples_rows_csv, normalize=None, sample_in_row=True, data_file_type='openms')
        # test we get the MZ and RT correct
        self.assertIn('MZ', exp.feature_metadata)
        self.assertIn('RT', exp.feature_metadata)
        self.assertAlmostEqual(exp.feature_metadata['MZ'].iloc[1], 118.0869)
        self.assertAlmostEqual(exp.feature_metadata['RT'].iloc[1], 23.9214)

    def test_read_qiime2(self):
        # test the non-hashed table
        exp = ca.read_qiime2(self.qiime2table, normalize=None, min_reads=None)
        self.assertEqual(exp.shape, (104, 658))

        # and the hashed table with rep seqs and taxonomy files
        exp = ca.read_qiime2(self.q2_cfs_table, sample_metadata_file=self.q2_cfs_map,
                             rep_seq_file=self.q2_cfs_repseqs, taxonomy_file=self.q2_cfs_taxonomy,
                             normalize=None, min_reads=None)
        self.assertEqual(exp.shape, (87, 2130))
        # test if the index is indeed sequences, and taxonomy is loaded correctly
        test_seq = 'TACGTAGGGAGCAAGCGTTGTCCGGAATTACTGGGTGTAAAGGGTGCGTAGGCGGGTATGCAAGTCATATGTGAAATACCGGGGCTCAACTCCGGGGCTGCATAAGAAACTGTATATCTTGAGTACAGGAGAGGTAAGCGGAATTCCTAG'
        self.assertEqual(exp.feature_metadata['Taxon'][test_seq], 'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Ruminococcaceae; g__; s__')

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
        for chead, cseq in _iter_fasta(f):
            seqs.append(cseq)
        self.assertCountEqual(seqs, exp.feature_metadata.index.values)
        shutil.rmtree(d)

    def test_save_biom(self):
        # NOTE: Currently not testing the save biom hdf with taxonomy
        # as there is a bug there!
        exp = ca.read_amplicon(self.test1_biom, self.test1_samp, normalize=None, min_reads=None)
        d = mkdtemp()
        f = join(d, 'test1.save.biom')
        # test the json biom format
        exp.save_biom(f, fmt='hdf5')
        newexp = ca.read_amplicon(f, self.test1_samp, normalize=None, min_reads=None)
        assert_experiment_equal(newexp, exp)
        # test the txt biom format
        exp.save_biom(f, fmt='txt')
        newexp = ca.read_amplicon(f, self.test1_samp, normalize=None, min_reads=None)
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
