# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
from scipy import sparse

from calour._testing import Tests, assert_experiment_equal
from calour.util import _convert_axis_name
import calour as ca


class ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)

    def test_record_sig(self):
        def foo(exp, axis=1, inplace=True):
            return exp

        ca.Experiment.foo = ca.Experiment._record_sig(foo)
        self.test1.foo()
        self.test1.foo()
        self.assertListEqual(
            self.test1._call_history,
            ['ExperimentTests.test_record_sig.<locals>.foo()'] * 2)

    def test_convert_axis_name_other_func(self):
        def foo(exp, inplace=True):
            return inplace
        ca.Experiment.foo = _convert_axis_name(foo)
        self.assertEqual(self.test1.foo(), True)

    def test_convert_axis_name(self):
        def foo(exp, axis=1, inplace=True):
            return axis, inplace

        ca.Experiment.foo = _convert_axis_name(foo)

        for i in (0, 's', 'sample', 'samples'):
            obs = self.test1.foo(axis=i)
            self.assertEqual(obs, (0, True))
            obs = self.test1.foo(i, inplace=False)
            self.assertEqual(obs, (0, False))

        for i in (1, 'f', 'feature', 'features'):
            obs = self.test1.foo(axis=i)
            self.assertEqual(obs, (1, True))
            obs = self.test1.foo(i, inplace=False)
            self.assertEqual(obs, (1, False))

        obs = self.test1.foo()
        self.assertEqual(obs, (1, True))

    def test_reorder_samples(self):
        # keep only samples S6 and S5
        new = self.test1.reorder([5, 4], axis=0)

        self.assertEqual(new.data.shape[0], 2)
        self.assertEqual(new.data.shape[1], self.test1.data.shape[1])

        # test sample_metadata are correct
        self.assertEqual(new.sample_metadata['id'][0], 6)
        self.assertEqual(new.sample_metadata['id'][1], 5)

        # test data are correct
        fid = 'GG'
        fpos = new.feature_metadata.index.get_loc(fid)
        self.assertEqual(new.data[0, fpos], 600)
        self.assertEqual(new.data[1, fpos], 500)

    def test_reorder_features_inplace(self):
        # test inplace reordering of features
        new = self.test1.reorder([2, 0], axis=1, inplace=True)
        fid = 'AG'
        fpos = self.test1.feature_metadata.index.get_loc(fid)
        self.assertIs(new, self.test1)
        self.assertEqual(new.data[0, fpos], 1)
        self.assertEqual(new.data[1, fpos], 2)

    def test_reorder_round_trip(self):
        # test double permuting of a bigger data set
        exp = ca.read(self.timeseries_biom, self.timeseries_samp, normalize=None)

        rand_perm_samples = np.random.permutation(exp.data.shape[0])
        rand_perm_features = np.random.permutation(exp.data.shape[1])
        rev_perm_samples = np.argsort(rand_perm_samples)
        rev_perm_features = np.argsort(rand_perm_features)
        new = exp.reorder(rand_perm_features, axis=1, inplace=False)
        new.reorder(rand_perm_samples, axis=0, inplace=True)
        new.reorder(rev_perm_features, axis=1, inplace=True)
        new.reorder(rev_perm_samples, axis=0, inplace=True)

        assert_experiment_equal(new, exp)

    def test_copy_experiment(self):
        exp = copy(self.test1)
        assert_experiment_equal(exp, self.test1)
        self.assertIsNot(exp, self.test1)

    def test_deep_copy_experiment(self):
        exp = deepcopy(self.test1)
        assert_experiment_equal(exp, self.test1)
        self.assertIsNot(exp, self.test1)

    def test_copy(self):
        exp = self.test1.copy()
        assert_experiment_equal(exp, self.test1)
        self.assertIsNot(exp, self.test1)
        # make sure it is a deep copy - not sharing the data
        exp.data[0, 0] = exp.data[0, 0] + 1
        self.assertNotEqual(exp.data[0, 0], self.test1.data[0, 0])

    def test_get_data_default(self):
        # default - do not modify the data
        exp = deepcopy(self.test1)
        data = exp.get_data()
        self.assertTrue(sparse.issparse(data))
        self.assertEqual(data.sum(), exp.data.sum())
        # test it's not a copy but inplace
        self.assertIs(data, exp.data)

    def test_get_data_copy(self):
        # lets force it to copy
        exp = deepcopy(self.test1)
        data = exp.get_data(copy=True)
        self.assertTrue(sparse.issparse(data))
        self.assertEqual(data.sum(), exp.data.sum())
        # test it's a copy but inplace
        self.assertIsNot(data, exp.data)

    def test_get_data_non_sparse(self):
        # force non-sparse, should copy
        exp = deepcopy(self.test1)
        data = exp.get_data(sparse=False)
        self.assertFalse(sparse.issparse(data))
        self.assertEqual(data.sum(), exp.data.sum())
        # test it's a copy but inplace
        self.assertIsNot(data, exp.data)

    def test_get_data_sparse(self):
        # force sparse, should not copy
        exp = deepcopy(self.test1)
        data = exp.get_data(sparse=True)
        self.assertTrue(sparse.issparse(data))
        self.assertEqual(data.sum(), exp.data.sum())
        # test it's not a copy but inplace
        self.assertIs(data, exp.data)

    def test_get_data_sparse_copy(self):
        # force sparse on a non-sparse matrix
        exp = deepcopy(self.test1)
        exp.sparse = False
        data = exp.get_data(sparse=True)
        self.assertTrue(sparse.issparse(data))
        self.assertEqual(data.sum(), exp.data.sum())
        # test it's a copy but inplace
        self.assertIsNot(data, exp.data)

    def test_to_pandas_dense(self):
        df = self.test1.to_pandas(sparse=False)
        data = self.test1.get_data(sparse=False)
        self.assertIsInstance(df, pd.DataFrame)
        npt.assert_array_almost_equal(df.values, data)

    def test_to_pandas_sparse(self):
        df = self.test1.to_pandas(sparse=True)
        data = self.test1.get_data(sparse=False)
        self.assertIsInstance(df, pd.SparseDataFrame)
        npt.assert_array_almost_equal(df.to_dense().values, data)

    def test_from_pands(self):
        df = self.test1.to_pandas(sparse=False)
        res = ca.Experiment.from_pandas(df)
        self.assertIsInstance(res, ca.Experiment)
        npt.assert_array_equal(res.feature_metadata.index.values, self.test1.feature_metadata.index.values)
        npt.assert_array_equal(res.sample_metadata.index.values, self.test1.sample_metadata.index.values)
        npt.assert_array_equal(res.get_data(sparse=False), self.test1.get_data(sparse=False))

    def test_from_pandas_with_experiment(self):
        df = self.test1.to_pandas(sparse=False)
        res = ca.Experiment.from_pandas(df, self.test1)
        assert_experiment_equal(res, self.test1)

    def test_from_pandas_reorder(self):
        df = self.test1.to_pandas(sparse=False)
        # let's reorder the dataframe
        df = df.sort_values(self.test1.feature_metadata.index.values[10])
        df = df.sort_values(df.index.values[0], axis=1)
        res = ca.Experiment.from_pandas(df, self.test1)
        # we need to reorder the original experiment
        exp = self.test1.sort_by_data(subset=[10], key=np.mean)
        exp = exp.sort_by_data(subset=[0], key=np.mean, axis=1)
        assert_experiment_equal(res, exp)

    def test_from_pandas_round_trip(self):
        data = np.array([[1, 2], [3, 4]])
        df = pd.DataFrame(data, index=['s1', 's2'], columns=['AAA', 'CCC'], copy=True)
        exp = ca.Experiment.from_pandas(df)
        res = exp.to_pandas()
        pdt.assert_frame_equal(res, df)

    def test_getitem(self):
        self.assertEqual(self.test1['S5', 'AG'], 5)
        self.assertEqual(self.test1['S4', 'AC'], 0)
        with self.assertRaises(KeyError):
            self.test1['Pita', 'AG']
        with self.assertRaises(KeyError):
            self.test1['S5', 'Pita']
        with self.assertRaises(SyntaxError):
            self.test1['S5']

    def test_shape(self):
        self.assertEqual(self.test1.shape, (21, 12))

    def test_getitem_slice(self):
        # 1st sample
        npt.assert_array_equal(self.test1['S1', :], self.test1.data.toarray()[0, :])
        # 2nd feature
        npt.assert_array_equal(self.test1[:, 'AT'],
                               self.test1.data.toarray()[:, 1])

    def test_repr(self):
        self.assertEqual(repr(self.test1),
                         '''Experiment test1.biom
---------------------
data dimension: 21 samples, 12 features''')
# sample IDs: Index(['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11',
#        'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
#        'badsample'],
#       dtype='object', name='#SampleID')
# feature IDs: Index(['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 'GA', 'GT', 'GG',
#        'badfeature'],
#       dtype='object')''')


if __name__ == "__main__":
    main()
