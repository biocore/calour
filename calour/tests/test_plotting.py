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

from calour import Experiment
from calour._testing import Tests


class PlotTests(Tests):
    def test_plot_hist(self):
        data = np.array([[0, 1], [2, 3]])
        exp = Experiment(data,
                         pd.DataFrame({'A': ['ab', 'cd'], 'B': ['ef', 'gh']}))
        counts, bins, fig = exp.plot_hist(bins=4)
        assert_array_almost_equal(counts, np.array([1] * 4))
        assert_array_almost_equal(bins, np.array([0., 0.75, 1.5, 2.25, 3.]))
        ax = fig.gca()
        # test the numbers on top of the histogram bars are correct
        self.assertEqual([i.get_text() for i in ax.texts], ['1'] * 4)

    def test_plot_stacked_bar(self):
        exp = Experiment(np.array([[0, 1], [2, 3]]),
                         pd.DataFrame({'A': ['ab', 'cd'], 'B': ['ef', 'gh']},
                                      index=['s1', 's2']),
                         pd.DataFrame({'genus': ['bacillus', 'listeria']}))
        # bar width used in plot_staked_bar
        width = 0.95
        fig = exp.plot_stacked_bar(sample_color_bars='A', legend_field='genus', xtick=None)
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


if __name__ == '__main__':
    main()
