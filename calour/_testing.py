# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase
from os.path import join, dirname, abspath
import logging

import pandas.testing as pdt
import numpy.testing as npt

import calour as ca


class Tests(TestCase):
    def setUp(self):
        # disable logging; otherwise, the tests will print all the
        # logging in the functions
        logging.disable(logging.CRITICAL)

        test_data_dir = join(dirname(abspath(__file__)), 'tests', 'data')
        self.test_data_dir = test_data_dir
        # a simple artificial biom table 21 sample x 12 feature
        self.test1_biom = join(test_data_dir, 'test1.biom')
        self.test1_samp = join(test_data_dir, 'test1.sample')
        self.test1_feat = join(test_data_dir, 'test1.feature')
        # a simpler artificial data set 9 sample x 8 feature
        self.test2_biom = join(test_data_dir, 'test2.biom')
        self.test2_samp = join(test_data_dir, 'test2.sample')
        self.test2_feat = join(test_data_dir, 'test2.feature')
        # a simple artificial biom table for paired testing
        self.test_paired_biom = join(test_data_dir, 'test_paired.biom')
        self.test_paired_samp = join(test_data_dir, 'test_paired.sample.txt')
        # a dense timeseries (real data)
        self.timeseries_biom = join(test_data_dir, 'timeseries.biom')
        self.timeseries_samp = join(test_data_dir, 'timeseries.sample')
        # a simple openms bucket table csv file
        self.openms_csv = join(test_data_dir, 'openms_bucket_table.csv')
        # a simple mzmine2 output table
        self.mzmine2_csv = join(test_data_dir, 'mzmine2_table.csv')
        # a simple mzmine2 output table with sampleids containing additional info separated by '_'
        self.mzmine2_with_idstr_csv = join(test_data_dir, 'mzmine2_table_with_idstr.csv')
        # a simple openms bucket table csv file with samples as rows
        self.openms_samples_rows_csv = join(test_data_dir, 'openms_bucket_table_samples_rows.csv')
        # a simple gnps data file for ms1 test data
        self.ms1_gnps = join(test_data_dir, 'ms1.gnps.txt')
        # the gnps exported data table
        self.gnps_table = join(test_data_dir, 'gnps_table.txt')
        # a metabolomics biom table with MZ_RT in feature id. linked to same gnps_clusterinfo file as the gnps_table
        self.ms_biom_table = join(test_data_dir, 'ms_biom_table.txt')
        # the gnps exported mapping file
        self.gnps_map = join(test_data_dir, 'gnps_map.txt')
        # the gnps per-metabolite info table (from clusterinfosummarygroup_attributes_withIDs_arbitraryattri  butes/XXX.tsv)
        self.gnps_cluster_info = join(test_data_dir, 'gnps_clusterinfosummarygroup.txt')
        # a fasta file for testing the AmpliconExperiment
        self.seqs1_fasta = join(test_data_dir, 'seqs1.fasta')
        # a qiime2 non-hashed biom table artifact
        self.qiime2table = join(test_data_dir, 'feature-table.qza')
        # a qiime2 dataset with hashed biom table, rep-seqs and taxonomy
        self.q2_cfs_table = join(test_data_dir, 'cfs-table.qza')
        self.q2_cfs_map = join(test_data_dir, 'cfs-map.txt')
        self.q2_cfs_repseqs = join(test_data_dir, 'cfs-rep-seqs.qza')
        self.q2_cfs_taxonomy = join(test_data_dir, 'cfs-taxonomy.qza')
        # An experiment used to create the ratio experiment using from_exp()
        self.rat_pre_biom = join(test_data_dir, 'ratio_exp_pre_table.biom')
        self.rat_pre_samp = join(test_data_dir, 'ratio_exp_pre_sample_metadata.txt')
        # A ratio experiment table created from the ratio_pre experiment
        self.rat_biom = join(test_data_dir, 'ratio-exp.biom')
        self.rat_samp = join(test_data_dir, 'ratio-exp_sample_metadata.txt')

    def assert_experiment_equal(self, exp1, exp2, check_history=False, almost_equal=True,
                                ignore_md_fields=('_calour_original_abundance',)):
        '''Test if two experiments are equal

        Parameters
        ----------
        exp1 : Experiment
        exp2 : Experiment
        check_history : bool, optional
            False (default) to skip testing the command history, True to compare also the command history
        almost_equal : bool, optional
            True (default) to test for almost identical, False to test the data matrix for exact identity
        ignore_md_fields : tuple of str or None
            list of metadata fields to ignore in the comparison. Default is ignoring the original read count (when sample loaded)
        '''
        self.assertIsInstance(exp1, ca.Experiment, 'exp1 not a calour Experiment class')
        self.assertIsInstance(exp2, ca.Experiment, 'exp2 not a calour Experiment class')

        # test the metadata
        sample_columns = exp1.sample_metadata.columns.union(exp1.sample_metadata.columns)
        feature_columns = exp1.feature_metadata.columns.union(exp2.feature_metadata.columns)
        if ignore_md_fields is not None:
            for cignore in ignore_md_fields:
                if cignore in sample_columns:
                    sample_columns = sample_columns.delete(sample_columns.get_loc(cignore))
                if cignore in feature_columns:
                    feature_columns = feature_columns.delete(feature_columns.get_loc(cignore))
        self.assertEqual(len(sample_columns.difference(exp1.sample_metadata.columns)), 0)
        self.assertEqual(len(sample_columns.difference(exp2.sample_metadata.columns)), 0)
        self.assertEqual(len(feature_columns.difference(exp1.feature_metadata.columns)), 0)
        self.assertEqual(len(feature_columns.difference(exp2.feature_metadata.columns)), 0)

        pdt.assert_frame_equal(exp1.feature_metadata[feature_columns], exp2.feature_metadata[feature_columns])
        pdt.assert_frame_equal(exp1.sample_metadata[sample_columns], exp2.sample_metadata[sample_columns])

        # test the data
        if almost_equal:
            dat1 = exp1.get_data(sparse=False, copy=True)
            dat2 = exp2.get_data(sparse=False, copy=True)
            npt.assert_array_almost_equal(dat1, dat2)
        else:
            npt.assert_array_equal(exp1.data, exp2.data)
        if check_history:
            if not exp1._call_history == exp2._call_history:
                raise AssertionError('histories are different between exp1 and exp2')
