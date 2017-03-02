# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
import numpy as np
from numpy.testing import assert_array_almost_equal

from matplotlib import pyplot as plt

import calour as ca
from calour._testing import Tests
from calour.heatmap.heatmap import ax_color_bar


class PlotTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test2_biom, self.test2_samp, self.test2_feat)

    def test_create_plot_gui(self):
        row, col = 1, 2
        for gui in ('cli', 'qt5', 'jupyter'):
            obs = self.test1.create_plot_gui(gui=gui, databases=[])
            obs.current_select = row, col
            sid, fid, abd, annt = obs.get_info()
            self.assertListEqual(annt, [])
            self.assertEqual(abd, self.test1.data[row, col])
            self.assertEqual(sid, self.test1.sample_metadata.index[row])
            self.assertEqual(fid, self.test1.feature_metadata.index[col])

    def test_heatmap(self):
        fig = self.test1.heatmap(sample_field='group',
                                 feature_field='ph',
                                 yticklabels_max=None,
                                 transform=None)
        ax = fig.gca()
        obs_images = ax.images
        # test only one heatmap exists
        self.assertEqual(len(obs_images), 1)
        # test heatmap is correct
        assert_array_almost_equal(self.test1.get_data(sparse=False).transpose(),
                                  obs_images[0].get_array())
        obs_lines = ax.lines
        # test only one line exists
        self.assertEqual(len(obs_lines), 1)
        # test the axvline is correct
        assert_array_almost_equal(obs_lines[0].get_xdata(),
                                  np.array([6.5, 6.5]))
        # test axis labels
        self.assertEqual(ax.xaxis.label.get_text(), 'group')
        self.assertEqual(ax.yaxis.label.get_text(), 'ph')
        # test axis tick labels
        obs_xticklabels = [i.get_text() for i in ax.xaxis.get_ticklabels()]
        self.assertListEqual(obs_xticklabels, ['1', '2'])
        obs_yticklabels = [i.get_text() for i in ax.yaxis.get_ticklabels()]
        self.assertListEqual(obs_yticklabels,
                             self.test1.feature_metadata['ph'].astype(str).tolist())

    def test_ax_color_bar(self):
        fig, ax = plt.subplots()
        colors = [(1.0, 0.0, 0.0, 1), (0.0, 0.5, 0.0, 1)]
        axes = ax_color_bar(ax, ['a', 'a', 'b'], 0.3, 0, colors)
        self.assertIs(ax, axes)
        # test face color rectangle in the bar
        self.assertListEqual(
            [i.get_facecolor() for i in axes.patches],
            colors)
        # test the position rectangle in the bar
        self.assertListEqual(
            [i.get_xy() for i in axes.patches],
            [(-0.5, 0), (1.5, 0)])
        # test the texts in each rectangle in the bar
        self.assertListEqual(
            [i.get_text() for i in axes.texts],
            ['a', 'b'])


if __name__ == '__main__':
    main()
