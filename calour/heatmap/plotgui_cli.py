# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from .plotgui import PlotGUI
from matplotlib import pyplot as plt


class PlotGUI_CLI(PlotGUI):
    '''Show the plot and relevant info in terminal

    It uses ``matplotlib`` only to display the plot and prints info on the
    terminal screen.
    '''
    def show_info(self):
        if 'taxonomy' in self.exp.feature_metadata:
            cname = self.exp.feature_metadata['taxonomy'][self.current_select[1]]
        # sample_name = self.exp.sample_metadata['#SampleID'][self.select_sample]
        # sample_name = self.exp.get_sample_md().iloc[self.select_sample]
        else:
            cname = self.exp.feature_metadata.index[self.current_select[1]]
        print(cname)
        return cname

    def __call__(self):
        '''Run the GUI.'''
        super().__call__()
        plt.show()
