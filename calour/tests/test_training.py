# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from os.path import join

from numpy.testing import assert_array_equal
import pandas as pd
import pandas.util.testing as pdt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

import calour as ca
from calour._testing import Tests
from calour.training import plot_cm, plot_roc


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

    def test_classify(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        d = dict(enumerate(iris.target_names))
        smd = pd.DataFrame({'plant': y}).replace(d)
        exp = ca.Experiment(X, smd, sparse=False)
        run = exp.classify('plant', KNeighborsClassifier,
                           predict='predict_proba',
                           cv=StratifiedKFold(3, random_state=0))
        res = next(run)
        obs = pd.read_table(join(self.test_data_dir, 'iris_result.txt'), index_col=0)
        pdt.assert_frame_equal(res, obs)
        # plot_roc(res)
        # from matplotlib import pyplot as plt
        # plt.show()

    def test_plot_roc_multi(self):
        result = pd.read_table(join(self.test_data_dir, 'iris_pred.txt'))
        ax = plot_roc(result)
        legend = ax.get_legend()
        for exp, obs in zip(legend.get_texts(),
                            ['Luck',
                             'versicolor (0.95 $\\pm$ 0.06)',
                             'virginica (0.96 $\\pm$ 0.04)',
                             'setosa (0.98 $\pm$ 0.01)']):
            self.assertEqual(exp.get_text(), obs)
        # from matplotlib import pyplot as plt
        # plt.show()

    def test_plot_roc_binary(self):
        result = pd.read_table(join(self.test_data_dir, 'iris_pred.txt'))
        result['Y_TRUE'] = ['setosa' if i == 'setosa' else 'not setosa'
                            for i in result['Y_TRUE']]
        result['not setosa'] = 1 - result['setosa']
        ax = plot_roc(result, pos_label='setosa')
        # from matplotlib import pyplot as plt
        # plt.show()
        legend = ax.get_legend()
        for exp, obs in zip(legend.get_texts(),
                            ['Luck',
                             'setosa (0.98 $\pm$ 0.01)']):
            self.assertEqual(exp.get_text(), obs)

    def test_plot_cm(self):
        result = pd.read_table(join(self.test_data_dir, 'iris_pred.txt'))
        ax = plot_cm(result)
        # from matplotlib import pyplot as plt
        # plt.show()
        obs = [((0, 0), '11'), ((1, 0), '1'), ((2, 0), '0'),
               ((0, 1), '1'), ((1, 1), '11'), ((2, 1), '0'),
               ((0, 2), '1'), ((1, 2), '1'), ((2, 2), '10')]
        for exp, obs in zip(ax.get_children(), obs):
            self.assertEqual(exp.get_text(), obs[1])
            self.assertEqual(exp.get_position(), obs[0])


if __name__ == "__main__":
    main()
