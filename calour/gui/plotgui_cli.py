from calour.gui.plotgui import PlotGUI


class PlotGUI_CLI(PlotGUI):
    '''CLI version of plot window GUI using print() to show info
    '''
    def update_info(self):
        tax_name = self.exp.feature_metadata['taxonomy'][self.select_feature]
        sample_name = self.exp.sample_metadata['DAY'][self.select_sample]
        # sample_name = self.exp.get_sample_md().iloc[self.select_sample]
        print(sample_name)
        print(tax_name)
