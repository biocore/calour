# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from .plotgui import PlotGUI
from .._doc import ds


class PlotGUI_CLI(PlotGUI):
    '''Show the plot and relevant info in terminal

    It uses ``matplotlib`` only to display the plot and prints info on the
    terminal screen.
    '''
    @ds.with_indent(8)
    def __init__(self, **kwargs):
        '''Init the GUI using the cli GUI.

        This GUI only enables the zooming/panning and displays information following a mouse click
        in the terminal window

        Keyword Arguments
        -----------------
        %(PlotGUI.parameters)s
        '''
        super().__init__(**kwargs)
        # create the figure to plot the heatmap into
        self._set_figure(None, kwargs['tree_size'])

    def __call__(self):
        '''Run the GUI.'''
        from matplotlib import pyplot as plt

        super().__call__()
        plt.show()
