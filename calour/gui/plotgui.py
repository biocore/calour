from logging import getLogger

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


logger = getLogger(__name__)


class PlotGUI:
    '''The base class for heatmap GUI

    Various gui classes inherit this class
    '''
    def __init__(self, exp, zoom_scale=2, scroll_offset=0):
        '''Init the gui windows class

        Store the experiment and the gui
        '''
        # the Experiment being plotted
        self.exp = exp

        # how much zooming in on key press
        self.zoom_scale = zoom_scale

        # how much to scroll on key press
        self.scroll_offset = scroll_offset

        # list of selected features
        self.selected_features = {}

        # list of selected samples
        self.selected_samples = {}

    def get_figure(self, newfig=None):
        ''' Get the figure to plot the heatmap into

        newfig: None or mpl.figure or False (optional)
            if None (default) create a new mpl figure.
            if mpl.figure use this figure to plot into.
            if False, use current figure to plot into.
        '''
        if newfig is None:
            fig = plt.figure()
        elif isinstance(newfig, mpl.figure.Figure):
            fig = newfig
        else:
            fig = plt.gcf()
        return fig

    def connect_functions(self, fig):
        '''Connect to the matplotlib callbacks for key and mouse '''
        self.canvas = fig.canvas
        self.canvas.mpl_connect('scroll_event', lambda f: _scroll_callback(f, hdat=self))
        self.canvas.mpl_connect('key_press_event', lambda f: _key_press_callback(f, hdat=self))
        self.canvas.mpl_connect('button_press_event', lambda f: _button_press_callback(f, hdat=self))

    def update_info(self):
        '''Update info when a new feature/sample is selected'''
        raise NotImplemented()

    def clear_selection(self):
        ''' Delete all shown selection lines '''
        for cline in self.selected_samples.values():
            self.axis.lines.remove(cline)
        self.selected_samples = {}
        for cline in self.selected_features.values():
            self.axis.lines.remove(cline)
        self.selected_features = {}

    def update_selection(self, samplepos=[], featurepos=[], toggle=True):
        '''Update the selection lines displayed

        Parameters
        ----------
        samplepos : list of int (optional)
            positions of samples to be added
        featurepos : list of int (optional)
            positions of features to be added
        toggle: bool (optional)
            True (default) to remove lines in the lists that are already selected, False to ignore
        '''
        for cpos in samplepos:
            if cpos not in self.selected_samples:
                self.selected_samples[cpos] = self.axis.axvline(x=cpos, color='white', linestyle='dotted')
            else:
                if toggle:
                    self.axis.lines.remove(self.selected_samples[cpos])
                    del self.selected_samples[cpos]
        for cpos in featurepos:
            if cpos not in self.selected_features:
                self.selected_features[cpos] = self.axis.axhline(y=cpos, color='white', linestyle='dotted')
            else:
                if toggle:
                    self.axis.lines.remove(self.selected_features[cpos])
                    del self.selected_features[cpos]
        self.canvas.draw()


def _button_press_callback(event, hdat):
    rx = int(round(event.xdata))
    ry = int(round(event.ydata))

    if event.key is None:
        # add selection to list
        hdat.clear_selection()
        hdat.update_selection(samplepos=[rx], featurepos=[ry])
    elif event.key == 'shift':
        if hdat.last_select_feature > ry:
            newbact = np.arange(hdat.last_select_feature, ry - 1, -1)
        else:
            newbact = np.arange(hdat.last_select_feature, ry + 1, 1)
        hdat.clear_selection()
        hdat.update_selection(featurepos=newbact)
    elif event.key == 'super':
        hdat.update_selection(featurepos=[ry])

    # store the latest selected feature/sample
    hdat.last_select_feature = ry
    hdat.last_select_sample = rx
    # and update the selected info
    hdat.update_info()


def _key_press_callback(event, hdat):
    ax = event.inaxes
    ylim_lower, ylim_upper = ax.get_ylim()
    xlim_lower, xlim_upper = ax.get_xlim()

    # set the scroll offset
    if hdat.scroll_offset > 0:
        x_offset = 1
        y_offset = 1
    else:
        x_offset = xlim_upper - xlim_lower
        y_offset = ylim_upper - ylim_lower

    if event.key == 'shift+up' or event.key == '=':
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) / hdat.zoom_scale)
    elif event.key == 'shift+down' or event.key == '-':
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) * hdat.zoom_scale)
    elif event.key == 'shift+right' or event.key == '+':
        ax.set_xlim(
            xlim_lower,
            xlim_lower + (xlim_upper - xlim_lower) / hdat.zoom_scale)
    elif event.key == 'shift+left' or event.key == '_':
        ax.set_xlim(
            xlim_lower,
            xlim_lower + (xlim_upper - xlim_lower) * hdat.zoom_scale)
    elif event.key == 'down':
        ax.set_ylim(ylim_lower - y_offset, ylim_upper - y_offset)
    elif event.key == 'up':
        ax.set_ylim(ylim_lower + y_offset, ylim_upper + y_offset)
    elif event.key == 'left':
        ax.set_xlim(xlim_lower - x_offset, xlim_upper - x_offset)
    elif event.key == 'right':
        ax.set_xlim(xlim_lower + x_offset, xlim_upper + x_offset)
    elif event.key == '.':
        hdat.last_select_feature += 1
        hdat.clear_selection()
        hdat.update_selection(samplepos=hdat.selected_samples, featurepos=[hdat.last_select_feature])
        hdat.update_info()
    elif event.key == ',':
        hdat.last_select_feature -= 1
        hdat.clear_selection()
        hdat.update_selection(samplepos=hdat.selected_samples, featurepos=[hdat.last_select_feature])
        hdat.update_info()
    elif event.key == '>':
        hdat.last_select_sample += 1
        hdat.clear_selection()
        hdat.update_selection(featurepos=hdat.selected_features, samplepos=[hdat.last_select_sample])
        hdat.update_info()
    elif event.key == '<':
        hdat.last_select_sample -= 1
        hdat.clear_selection()
        hdat.update_selection(featurepos=hdat.selected_features, samplepos=[hdat.last_select_sample])
        hdat.update_info()

    else:
        return

#    plt.tight_layout()
    hdat.canvas.draw()


def _scroll_callback(event, hdat):
    ax = event.inaxes
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    xdata = event.xdata  # get event x location
    ydata = event.ydata  # get event y location
    x_left = xdata - cur_xlim[0]
    x_right = cur_xlim[1] - xdata
    y_top = ydata - cur_ylim[0]
    y_bottom = cur_ylim[1] - ydata
    if event.button == 'up':
        scale_factor = 1. / hdat.zoom_scale
    elif event.button == 'down':
        scale_factor = hdat.zoom_scale
    else:
        # deal with something that should never happen
        scale_factor = 1
        print(event.button)
    # set new limits
    ax.set_xlim([xdata - x_left * scale_factor,
                 xdata + x_right * scale_factor])
    ax.set_ylim([ydata - y_top * scale_factor,
                 ydata + y_bottom * scale_factor])

    hdat.canvas.draw()  # force re-draw
