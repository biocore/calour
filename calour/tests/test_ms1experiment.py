# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy.testing as npt

from calour._testing import Tests
import calour as ca


class ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read_amplicon(self.test1_biom, self.test1_samp,
                                      min_reads=1000, normalize=10000)

    def test_filter_mz_rt(self):
        # load an mzmine2 metabolomics table, and associated gnps clusterinfo file
        exp = ca.read_ms(self.mzmine2_csv, sample_metadata_file=self.gnps_map,
                         data_file_type='mzmine2', use_gnps_id_from_AllFiles=False, normalize=None)

        # mz filtering
        res = exp.filter_mz_rt(100)
        self.assertEqual(len(res.feature_metadata), 1)
        self.assertEqual(res.feature_metadata['MZ'].values, [100])

        res = exp.filter_mz_rt([100, 201])
        self.assertEqual(len(res.feature_metadata), 1)
        self.assertEqual(res.feature_metadata['MZ'].values, [100])

        res = exp.filter_mz_rt([100, 201], mz_tolerance=1)
        self.assertEqual(len(res.feature_metadata), 2)
        npt.assert_array_equal(res.feature_metadata['MZ'].values, [100, 200])

        res = exp.filter_mz_rt([100, 201], negate=True)
        self.assertEqual(len(res.feature_metadata), 5)

        # rt filtering
        res = exp.filter_mz_rt(rt=[1, 2.5])
        self.assertEqual(len(res.feature_metadata), 1)
        self.assertEqual(res.feature_metadata['RT'].values, [1])

        res = exp.filter_mz_rt(rt=[1, 2.5], rt_tolerance=0.5)
        self.assertEqual(len(res.feature_metadata), 3)
        npt.assert_array_equal(res.feature_metadata['RT'].values, [1, 2, 3])

        # complex - both mz and rt
        res = exp.filter_mz_rt([101, 200, 400, 505], [1, 3, 4, 5], mz_tolerance=2)
        self.assertEqual(res.shape[1], 2)

    def test_get_bad_features(self):
        # load an mzmine2 metabolomics table, and associated gnps clusterinfo file
        exp = ca.read_ms(self.mzmine2_csv, sample_metadata_file=self.gnps_map,
                         data_file_type='mzmine2', use_gnps_id_from_AllFiles=False, normalize=None)
        # get rid of the all 0s metabolite (to get rid of std=0 warning)
        exp = exp.filter_sum_abundance(0.1)

        res = exp.get_bad_features()
        # no samples filtered away
        self.assertEqual(res.shape[0], 6)
        # default parameters don't identify and suspicious features
        self.assertEqual(res.shape[1], 0)

        res = exp.get_bad_features(mz_tolerance=100, rt_tolerance=0.5)
        self.assertEqual(res.shape[1], 0)

        res = exp.get_bad_features(rt_tolerance=1)
        self.assertEqual(res.shape[1], 0)

        res = exp.get_bad_features(mz_tolerance=100, rt_tolerance=1)
        self.assertEqual(res.shape[1], 2)

        res = exp.get_bad_features(mz_tolerance=100, rt_tolerance=1, corr_thresh=0.2)
        self.assertEqual(res.shape[1], 4)

    def test_merge_similar_features(self):
        # load an mzmine2 metabolomics table, and associated gnps clusterinfo file
        exp = ca.read_ms(self.mzmine2_csv, sample_metadata_file=self.gnps_map,
                         data_file_type='mzmine2', use_gnps_id_from_AllFiles=False, normalize=None)
        # no merging since features are far away
        res = exp.merge_similar_features()
        self.assertEqual(res.shape[1], 6)

        # a little merging
        res = exp.merge_similar_features(mz_tolerance=100, rt_tolerance=1)
        self.assertEqual(res.shape[1], 3)
        self.assertEqual(res.feature_metadata.at[85022, '_calour_merge_ids'], '85022;93277')

        # a lot of merging
        res = exp.merge_similar_features(mz_tolerance=400, rt_tolerance=6)
        self.assertEqual(res.shape[1], 2)
        self.assertEqual(res.feature_metadata.at[121550, '_calour_merge_ids'], '121550')
