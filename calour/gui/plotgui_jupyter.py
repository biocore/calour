import matplotlib
from matplotlib.figure import Figure

from ipywidgets import *
from IPython.display import display, clear_output

from calour.bactdb import BactDB
from calour.gui.plotgui import PlotGUI


# app_ref=set()


class PlotGUI_Jupyter(PlotGUI):
    '''QT5 version of plot winfow GUI

    We open the figure as a widget inside the qt5 window
    '''
    def __init__(self, *kargs, **kwargs):
        PlotGUI.__init__(self, *kargs, **kwargs)
        self.bactdb = BactDB()

    def get_figure(self, newfig=None):
        fig = PlotGUI.get_figure(self, newfig=newfig)
        self.labtax = Label('Feature:')
        self.labsamp = Label('Sample')
        self.labreads = Label('Reads:')
        self.labdb = HTML('?')
        self.labdb.layout.overflow = 'auto'
        self.labdb.layout.overflow_x = 'auto'
        self.labdb.layout.max_height = '50px'
        self.labdb.layout.white_space = 'nowrap'
        # self.labdb.layout.width = '200px'
        self.zoomin = Button(description='+')
        self.zoomin.on_click(lambda f: zoom_in(f, self))
        display(self.zoomin)
        display(self.labtax)
        display(self.labsamp)
        display(self.labreads)
        display(self.labdb)
        return fig

    def update_info(self):
        taxname = self.exp.feature_metadata['taxonomy'][self.select_feature]
        sampname = self.exp.sample_metadata.index[self.select_sample]
        sequence = self.exp.feature_metadata.index[self.select_feature]
        self.labtax.value = 'Feature: %s' % taxname
        self.labsamp.value = 'Sample: %s' % sampname
        self.labreads.value = 'Reads:{:.01f}'.format(self.exp.get_data()[self.select_sample, self.select_feature])
        info = self.bactdb.getannotationstrings(sequence)
        idata = ''
        for cinfo in info:
            cstr = cinfo[1]
            ccolor = self._get_color(cinfo[0])
            idata += '<p style="color:%s;white-space:nowrap;">%s</p>' % (ccolor, cstr)
        self.labdb.value = idata

    def _get_color(self, details):
        if details['annotationtype'] == 'diffexp':
            ccolor = 'blue'
        elif details['annotationtype'] == 'contamination':
            ccolor = 'red'
        elif details['annotationtype'] == 'common':
            ccolor = 'green'
        elif details['annotationtype'] == 'highfreq':
            ccolor = 'green'
        else:
            ccolor = 'black'
        return ccolor


def zoom_in(b, hdat):
    ax = hdat.fig.gca()
    ylim_lower, ylim_upper = ax.get_ylim()
    xlim_lower, xlim_upper = ax.get_xlim()
    ax.set_ylim(
        ylim_lower,
        ylim_lower + (ylim_upper - ylim_lower) / hdat.zoom_scale)
    hdat.canvas.draw()
    clear_output(wait=True)
    display(hdat.fig)

