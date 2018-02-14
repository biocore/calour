# ----------------------------------------------------------------------------
# Copyright (c) 2017--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

import calour as ca
from calour._testing import Tests
from calour.util import compute_prevalence
from calour.plotting import _compute_frac_nonzero


class PlotTests(Tests):
    def test_plot_hist(self):
        exp = ca.Experiment(np.array([[0, 1], [2, 3]]),
                            pd.DataFrame({'A': ['ab', 'cd'], 'B': ['ef', 'gh']}))
        counts, bins, ax = exp.plot_hist(bins=4)
        assert_array_almost_equal(counts, np.array([1] * 4))
        assert_array_almost_equal(bins, np.array([0., 0.75, 1.5, 2.25, 3.]))
        # test the numbers on top of the histogram bars are correct
        self.assertEqual([i.get_text() for i in ax.texts], ['1'] * 4)

    def test_plot_stacked_bar(self):
        exp = ca.Experiment(np.array([[0, 1], [2, 3]]),
                            pd.DataFrame({'A': ['ab', 'cd'], 'B': ['ef', 'gh']},
                                         index=['s1', 's2']),
                            pd.DataFrame({'genus': ['bacillus', 'listeria']}))
        # bar width used in plot_staked_bar
        width = 0.95
        fig = exp.plot_stacked_bar(sample_color_bars='A', field='genus', xtick=None)
        self.assertEqual(len(fig.axes), 3)
        # test bar ax
        ax = fig.axes[0]
        # test x axis tick labels
        obs_xticklabels = [i.get_text() for i in ax.xaxis.get_ticklabels()]
        self.assertListEqual(obs_xticklabels, ['s1', 's2'])
        # get all the bars
        bars = ax.get_children()[:4]
        xs = [0 - width / 2, 1 - width / 2]
        ys = [0, 0, 0, 2]
        heights = [0, 2, 1, 3]
        for i, bar in enumerate(bars):
            bbox = bar.get_bbox()
            # coordinate of lower left corner of the bar: x, y
            # width and height of the bar: w, h
            x, y, w, h = bbox.bounds
            # i % 2 because there are only columns
            self.assertAlmostEqual(x, xs[i % 2])
            self.assertAlmostEqual(w, width)
            self.assertAlmostEqual(y, ys[i])
            self.assertAlmostEqual(h, heights[i])

    def test_plot_abund_prevalence(self):
        self.test1 = ca.read(self.test1_biom, self.test1_samp, self.test1_feat, normalize=100)
        self.test1.sparse = False
        ax = self.test1.filter_samples('group', ['1', '2']).plot_abund_prevalence('group', min_abund=50)
        grp1 = self.test1.filter_samples('group', '1')
        grp2 = self.test1.filter_samples('group', '2')
        lines = ax.get_lines()
        # only 2 features passed min_abund
        self.assertEqual(len(lines), 2)
        # only one feature for each group
        mean_abund = grp1.data.sum(axis=0) / grp1.data.shape[0]
        self.assertEqual(np.sum(mean_abund > 50), 1)
        f = grp1.data[:, mean_abund > 50]
        x, y = compute_prevalence(f)
        assert_array_almost_equal(np.array([[i, j] for i, j in zip(x, y)]),
                                  lines[0].get_xydata())
        mean_abund = grp2.data.sum(axis=0) / grp2.data.shape[0]
        self.assertEqual(np.sum(mean_abund > 50), 1)
        f = grp2.data[:, mean_abund > 50]
        x, y = compute_prevalence(f)
        assert_array_almost_equal(np.array([[i, j] for i, j in zip(x, y)]),
                                  lines[1].get_xydata())

    def test_plot_core_features(self):
        np.random.seed(12345)
        self.test1 = ca.read(self.test1_biom, self.test1_samp, self.test1_feat, normalize=100)
        self.test1.sparse = False
        ax = self.test1.filter_samples(
            'group', ['1', '2']).plot_core_features(
                field='group', steps=(2, 12), iterations=2)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 6)

    def test_compute_frac_nonzero(self):
        data = np.array([[4, 5, 0, 3, 5, 1, 4, 3],
                         [0, 2, 1, 0, 5, 3, 1, 5],
                         [4, 0, 4, 3, 0, 2, 0, 5],
                         [2, 4, 0, 4, 2, 0, 1, 0],
                         [3, 3, 5, 3, 1, 0, 0, 1]])

        frac = _compute_frac_nonzero(data, [5, 3, 2], cutoff=0.1, frac=1, random_state=1)
        assert_array_almost_equal(frac, np.array([0, 0.25, 4/7]))

        frac = _compute_frac_nonzero(data, [5, 3, 2], cutoff=0.1, frac=0.00001, random_state=1)
        assert_array_almost_equal(frac, np.array([1, 1, 1]))

        frac = _compute_frac_nonzero(data, [5, 3, 2], cutoff=5, frac=1, random_state=1)
        assert_array_almost_equal(frac, np.array([0, 0, 0]))

    def test_plot_scatter_matrix(self):
        self.test2 = ca.read(self.test2_biom, self.test2_samp, self.test2_feat, normalize=100)
        fids = ['AA', 'AT', 'AG', 'AC']
        fig = self.test2.plot_scatter_matrix('ori.order', fids, ncols=2, nrows=2)
        self.assertEqual(len(fig.axes), 4)
        for ax, fid in zip(fig.axes, fids):
            # check the trend line
            self.assertEqual(len(ax.lines), 1)
            xobs = ax.lines[0].get_data()[0]
            xexp = self.test2.sample_metadata['ori.order'].values
            assert_array_almost_equal(xobs, xexp)
            yobs = ax.get_children()[0].get_offsets()[:, 1]
            yexp = self.test2[:, fid]
            assert_array_almost_equal(yobs, yexp)


if __name__ == '__main__':
    main()
