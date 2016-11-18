from calour.heatmap import PlotGUI


class PlotGUI_CLI(PlotGUI):
    '''CLI version of plot window GUI using print() to show info
    '''
    def update_info(self,taxname):
        print(taxname)
