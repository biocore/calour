# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt


logger = getLogger(__name__)


class PlotGUI(ABC):
    '''abstract base class for heatmap GUI.

    Keys:


    Attributes
    ----------
    exp : ``Experiment``
        the experiment associated with this gui
    selected_features : dict of matplotlib.lines.Line2D
        used to track the selected features and plot horizontal lines for each selectiom
        keys - the selected features indeices, values - the line id for each selected feature
    selected_samples : dict of matplotlib.lines.Line2D
        used to track the selected samples and plot vertical lines for each selectiom
        keys - the selected sample indices, values - the line id for each selected samples
    current_select : tuple of (int, int)
        current selected point
    zoom_scale : numeric
        the scaling factor for zooming
    scroll_offset : numeric (optional)
        The amount of bacteria/samples to scroll when arrow key pressed
        0 (default) to scroll one full screen every keypress
        >0 : scroll than constant amount of bacteria per keypress
    figure : ``matplotlib.figure.Figure``
        The figure where the heatmap will be plotted into. It creates one by default.
    axis : matplotlib axes obtained from ``figure``.
    databases : list
        the database to interact with

    Parameters
    ----------
    exp :
    zoom_scale :
    scroll_offset :
    '''
    def __init__(self, exp, zoom_scale=2, scroll_offset=0, figure=None, databases=None):
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
        # current selected point
        self.current_select = None
        # list of databases to interface with
        if databases is None:
            self.databases = []
        # the default database used when annotating features
        self._annotation_db = None

        # create the figure to plot the heatmap into
        if figure is None:
            self.figure = plt.figure()
        else:
            self.figure = figure

    @property
    def axis(self):
        # this attr has to be property so it is updated on mouse/key events
        return self.figure.gca()

    def get_info(self):
        '''Get info for the selected feature/sample

        Returns
        -------
        tuple of (str, str, numeric, str or ``None``)
            sample id, feature id, abundance, taxonomy
        '''
        if 'taxonomy' in self.exp.feature_metadata:
            tax = self.exp.feature_metadata['taxonomy'][self.current_select[1]]
        else:
            tax = None

        fid = self.exp.feature_metadata.index[self.current_select[1]]
        sid = self.exp.sample_metadata.index[self.current_select[0]]
        abd = self.exp.data[self.current_select[0], self.current_select[1]]

        annt = []
        for cdatabase in self.databases:
            try:
                cannt = cdatabase.get_seq_annotation_strings(fid)
                if len(cannt) == 0:
                    cannt = [[{'annotationtype': 'not found'},
                              'No annotation found in database %s' % cdatabase.get_name()]]
                else:
                    for cannotation in cannt:
                        cannotation[0]['_db_interface'] = cdatabase
            except:
                cannt = 'error connecting to db %s' % cdatabase.get_name()
            annt.extend(cannt)

        return sid, fid, abd, tax, annt

    def show_info(self):
        print(self.get_info())

    def __call__(self):
        '''Run the GUI.'''
        self.connect_functions()

    def connect_functions(self):
        '''Connect to the matplotlib callbacks for key and mouse '''
        canvas = self.figure.canvas
        # comment out scroll event for now
        # canvas.mpl_connect('scroll_event', self.scroll_zoom_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)

    def scroll_zoom_callback(self, event):
        '''Zoom upon mouse scroll event'''
        logger.debug(repr(event))
        ax = event.inaxes
        # ax is None when scrolled outside the heatmap
        if ax is None:
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        x_left = xdata - cur_xlim[0]
        x_right = cur_xlim[1] - xdata
        y_top = ydata - cur_ylim[0]
        y_bottom = cur_ylim[1] - ydata
        if event.button == 'up':
            scale_factor = 1. / self.zoom_scale
        elif event.button == 'down':
            scale_factor = self.zoom_scale
        else:
            # deal with something that should never happen
            logger.warning('unknow scroll movement')
            return
        # set new limits
        ax.set_xlim([xdata - x_left * scale_factor,
                     xdata + x_right * scale_factor])
        ax.set_ylim([ydata - y_top * scale_factor,
                     ydata + y_bottom * scale_factor])

        self.figure.canvas.draw()  # force re-draw

    def button_press_callback(self, event):
        '''Select upon mouse button press.

        button only: empty the previous selection and select the current point

        button + shift: select all the features in the rectangle between
        current selection and last selecton.

        button + super: add current selected features to the selected list
        '''
        logger.debug(repr(event))
        ax = event.inaxes
        # ax is None when clicked outside the heatmap
        if ax is None:
            return

        rx = int(round(event.xdata))
        ry = int(round(event.ydata))

        if event.key is None:
            # add selection to list
            self.clear_selection()
            self.update_selection(samplepos=[rx], featurepos=[ry])
        elif event.key == 'shift':
            try:
                last_selected_feature = self.current_select[1]
            except IndexError:
                logger.critical('You have not selected any previously.')
                return
            if last_selected_feature > ry:
                features = np.arange(last_selected_feature, ry - 1, -1)
            else:
                features = np.arange(last_selected_feature, ry + 1, 1)
            self.clear_selection()
            self.update_selection(featurepos=features)
        elif event.key == 'super':
            self.update_selection(featurepos=[ry])

        self.current_select = (rx, ry)
        # and show the selected info
        self.show_info()

    def key_press_callback(self, event):
        '''Move/zoom upon key pressing.'''
        logger.debug('%r: %s key pressed' % (event, event.key))
        ax = event.inaxes
        if ax is None:
            return

        ylim_lower, ylim_upper = ax.get_ylim()
        xlim_lower, xlim_upper = ax.get_xlim()

        # set the scroll offset
        if self.scroll_offset > 0:
            x_offset = self.scroll_offset
            y_offset = self.scroll_offset
        else:
            x_offset = xlim_upper - xlim_lower
            y_offset = ylim_upper - ylim_lower

        if event.key == 'shift+up' or event.key == '=':
            ax.set_ylim(
                ylim_lower,
                ylim_lower + (ylim_upper - ylim_lower) / self.zoom_scale)
        elif event.key == 'shift+down' or event.key == '-':
            ax.set_ylim(
                ylim_lower,
                ylim_lower + (ylim_upper - ylim_lower) * self.zoom_scale)
        elif event.key == 'shift+right' or event.key == '+':
            ax.set_xlim(
                xlim_lower,
                xlim_lower + (xlim_upper - xlim_lower) / self.zoom_scale)
        elif event.key == 'shift+left' or event.key == '_':
            ax.set_xlim(
                xlim_lower,
                xlim_lower + (xlim_upper - xlim_lower) * self.zoom_scale)
        elif event.key == 'down':
            ax.set_ylim(ylim_lower - y_offset, ylim_upper - y_offset)
        elif event.key == 'up':
            ax.set_ylim(ylim_lower + y_offset, ylim_upper + y_offset)
        elif event.key == 'left':
            ax.set_xlim(xlim_lower - x_offset, xlim_upper - x_offset)
        elif event.key == 'right':
            ax.set_xlim(xlim_lower + x_offset, xlim_upper + x_offset)
        elif event.key in {'.', ',', '<', '>'}:
            shift = {'.': (0, 1),
                     ',': (0, -1),
                     '<': (-1, 0),
                     '>': (1, 0)}
            try:
                self.current_select = (self.current_select[0] + shift[event.key][0],
                                       self.current_select[1] + shift[event.key][1])
            except IndexError:
                logger.warning('You have not selected any previously.')
                return
            self.clear_selection()
            self.update_selection(
                samplepos=[self.current_select[0]], featurepos=[self.current_select[1]])
            self.show_info()
        else:
            logger.warning('Unrecoginzed key: %s' % event.key)
            return

        self.figure.canvas.draw()

    def clear_selection(self):
        '''Delete all shown selection lines '''
        for cline in self.selected_samples.values():
            self.axis.lines.remove(cline)
            logger.debug('remove sample selection %r' % cline)
        self.selected_samples = {}
        for cline in self.selected_features.values():
            self.axis.lines.remove(cline)
            logger.debug('remove sample selection %r' % cline)
        self.selected_features = {}

    def update_selection(self, samplepos=(), featurepos=(), toggle=True):
        '''Update the selection

        Parameters
        ----------
        samplepos : iterable of int (optional)
            positions of samples to be added
        featurepos : iterable of int (optional)
            positions of features to be added
        toggle: bool (optional)
            True (default) to remove lines in the lists that are already selected.
            False to ignore
        '''
        for cpos in samplepos:
            if cpos not in self.selected_samples:
                self.selected_samples[cpos] = self.axis.axvline(
                    x=cpos, color='white', linestyle='dotted')
                logger.debug('add sample selection %r' % cpos)
            else:
                if toggle:
                    self.axis.lines.remove(self.selected_samples[cpos])
                    del self.selected_samples[cpos]
        for cpos in featurepos:
            if cpos not in self.selected_features:
                self.selected_features[cpos] = self.axis.axhline(
                    y=cpos, color='white', linestyle='dotted')
                logger.debug('add sample selection %r' % cpos)
            else:
                if toggle:
                    self.axis.lines.remove(self.selected_features[cpos])
                    del self.selected_features[cpos]
        self.figure.canvas.draw()

    def get_selected_seqs(self):
        '''Get the list of selected sequences

        Parameters
        ----------

        Returns
        -------
        seqs : list of str
            The selected sequences ('ACGT')
        '''
        return self.exp.feature_metadata.index[self.selected_features.keys()]
