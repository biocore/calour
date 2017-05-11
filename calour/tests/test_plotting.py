# ----------------------------------------------------------------------------
# Copyright (c) 2017--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main, TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from calour import Experiment


class PlotTests(TestCase):
    def test_plot_hist(self):
        exp = Experiment(np.array([[0, 1], [2, 3]]),
                         pd.DataFrame({'A': ['ab', 'cd'], 'B': ['ef', 'gh']}))
        counts, bins, fig = exp.plot_hist(bins=4)
        assert_array_almost_equal(counts, np.array([1] * 4))
        assert_array_almost_equal(bins, np.array([0., 0.75, 1.5, 2.25, 3.]))
        ax = fig.gca()
        # test the numbers on top of the histogram bars are correct
        self.assertEqual([i.get_text() for i in ax.texts], ['1'] * 4)


if __name__ == '__main__':
    main()
