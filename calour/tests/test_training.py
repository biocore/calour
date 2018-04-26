# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

from numpy.testing import assert_array_equal
import pandas as pd

import calour as ca
from calour._testing import Tests


class TTests(Tests):
    def setUp(self):
        super().setUp()
        self.test2_sparse = ca.read(self.test2_biom, self.test2_samp, self.test2_feat, normalize=None)
        self.test2_dense = ca.read(self.test2_biom, self.test2_samp, self.test2_feat, sparse=False, normalize=None)

    def test_add_sample_metadata_as_features(self):
        new = self.test2_sparse.add_sample_metadata_as_features(['categorical'])
        dat = new.data.toarray()
        assert_array_equal(dat[:, 0:3],
                           [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 3)
        self.assertListEqual(new.feature_metadata.index[:3].tolist(),
                             ['categorical=A', 'categorical=B', 'categorical=C'])

    def test_add_sample_metadata_as_features_dense(self):
        new = self.test2_dense.add_sample_metadata_as_features(['categorical'])
        assert_array_equal(new.data[:, 0:3],
                           [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 3)
        self.assertListEqual(new.feature_metadata.index[:3].tolist(),
                             ['categorical=A', 'categorical=B', 'categorical=C'])

    def test_split_train_test(self):
        train_X, test_X, train_y, test_y = self.test2_dense.split_train_test(
            test_size=3, field='group', stratify='categorical', random_state=7)
        self.assertListEqual(test_y.tolist(), [1, 2, 1])
        self.assertListEqual(test_y.index.tolist(), ['S3', 'S8', 'S1'])
        self.assertListEqual(train_y.tolist(), [2, 1, 1, 1, 1, 1])
        self.assertListEqual(train_y.index.tolist(), ['S9', 'S6', 'S5', 'S2', 'S4', 'S7'])

    def test_classify_cv(self):
        from sklearn import datasets
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from matplotlib import pyplot as plt
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        d = dict(enumerate(iris.target_names))
        smd = pd.DataFrame({'plant': y}).replace(d)
        exp = ca.Experiment(X, smd)
        a, b = next(exp.classify_cv('plant', RandomForestClassifier, StratifiedKFold(n_splits=3)))
        plt.show()

if __name__ == "__main__":
    main()
