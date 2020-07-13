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

    def test_filter_mz(self):
        # load an mzmine2 metabolomics table, and associated gnps clusterinfo file
        exp = ca.read_ms(self.mzmine2_csv, sample_metadata_file=self.gnps_map,
                         data_file_type='mzmine2', use_gnps_id_from_AllFiles=False, normalize=None)

        res = exp.filter_mz(100)
        self.assertEqual(len(res.feature_metadata), 1)
        self.assertEqual(res.feature_metadata['MZ'].values, [100])

        res = exp.filter_mz([100, 201])
        self.assertEqual(len(res.feature_metadata), 1)
        self.assertEqual(res.feature_metadata['MZ'].values, [100])

        res = exp.filter_mz([100, 201], tolerance=1)
        self.assertEqual(len(res.feature_metadata), 2)
        npt.assert_array_equal(res.feature_metadata['MZ'].values, [100, 200])

        res = exp.filter_mz([100, 201], negate=True)
        self.assertEqual(len(res.feature_metadata), 5)

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
