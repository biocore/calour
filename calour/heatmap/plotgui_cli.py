# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from .plotgui import PlotGUI


class PlotGUI_CLI(PlotGUI):
    '''CLI version of plot window GUI using print() to show info '''
    def show_info(self):
        if 'taxonomy' in self.exp.feature_metadata:
            cname = self.exp.feature_metadata['taxonomy'][self.last_select_feature]
        # sample_name = self.exp.sample_metadata['#SampleID'][self.select_sample]
        # sample_name = self.exp.get_sample_md().iloc[self.select_sample]
        # print(sample_name)
        else:
            cname = self.exp.feature_metadata.index[self.last_select_feature]

        print(cname)
