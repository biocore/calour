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
from matplotlib.gridspec import GridSpec

from .._doc import ds

logger = getLogger(__name__)


@ds.get_sectionsf('PlotGUI')
class PlotGUI(ABC):
    '''abstract base class for heatmap GUI.

    The grid of subplots looks like this:

    +------------------------------------+---+----------------+
    |          sample color bar          |   |            |   |
    |------------------------------------+---+------------+---|
    |                                    | f |            |   |
    |                                    | e |            |   |
    |                                    | a |            | c |
    |                                    | t |            | o |
    |              heatmap               | u |            | l |
    |                                    | r |            | o |
    |                                    | e |            | r |
    |                                    |   |    tree    |   |
    |                                    | c | (optional  | l |
    |                                    | o |  column)   | e |
    |                                    | l |            | g |
    |                                    | o |            | e |
    |                                    | r |            | n |
    |                                    |   |            | d |
    |                                    | b |            |   |
    |                                    | a |            |   |
    |                                    | r |            |   |
    +------------------------------------+---+------------+---+

    Attributes
    ----------
    exp : Experiment
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
    scroll_offset : numeric, optional
        The amount of columns/rows to scroll when arrow key pressed
        0 (default) to scroll one full screen every keypress
        >0 : scroll than constant number of columns/rows per keypress
    figure : matplotlib.figure.Figure
        The figure where the heatmap and other axes will be plotted into.
    ax_hm : matplotlib.axes.Axes
        Axes for the heatmap.
    ax_sbar : matplotlib.axes.Axes
        Axes for the sample colorbar
    ax_fbar : matplotlib.axes.Axes
        Axes for the feature colorbar
    ax_tre : matplotlib.axes.Axes
        Axes for the dendrogram/tree
    ax_legend : matplotlib.axes.Axes
        Axes for the color legend
    databases : list on None, optional
        the databases to interact with.

    Parameters
    ----------
    exp : Experiment
        object associated with this GUI
    zoom_scale : float or int
        the scaling factor for zooming
    scroll_offset : float
        The amount of columns/rows to scroll when arrow key pressed
    databases : list of str or None
        the databases to interact with.
    tree_size : int (>= 0)
        the width of the axes to plot a tree. 7 is a good value to start.

    '''
    def __init__(self, exp, zoom_scale=2, scroll_offset=0, databases=None, tree_size=0):
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

    def _set_figure(self, figure=None, tree_size=0):
        import matplotlib.pyplot as plt
        if figure is None:
            self.figure = plt.figure()
        else:
            self.figure = figure
        if tree_size == 0:
            gs = GridSpec(2, 3, width_ratios=[12, 1, 0.5], height_ratios=[1, 12])
            hm_ax = self.figure.add_subplot(gs[3])
            self.ax_sbar = self.figure.add_subplot(gs[0], sharex=hm_ax)
            self.ax_sbar.axis('off')
            self.ax_fbar = self.figure.add_subplot(gs[4], sharey=hm_ax)
            self.ax_fbar.axis('off')
            self.ax_hm = hm_ax
            self.ax_legend = self.figure.add_subplot(gs[5])
            self.gridspec = gs
        else:
            gs = GridSpec(2, 4, width_ratios=[12, 1, tree_size, 0.5], height_ratios=[1, 12])
            hm_ax = self.figure.add_subplot(gs[4])
            self.ax_sbar = self.figure.add_subplot(gs[0], sharex=hm_ax)
            self.ax_sbar.axis('off')
            self.ax_fbar = self.figure.add_subplot(gs[5], sharey=hm_ax)
            self.ax_fbar.axis('off')
            self.ax_hm = hm_ax
            self.ax_tre = self.figure.add_subplot(gs[6], sharey=hm_ax)
            self.ax_tre.axis('off')
            self.ax_legend = self.figure.add_subplot(gs[7])
            self.gridspec = gs

    def resize_figure(self, *args, **kwargs):
        '''Resize the figure.

        It is a convenience function that wraps on
        ``matplotlib.Figure.set_size_inches``.

        Parameters
        ----------
        args, kwargs: tuple, dict
            arguments passing to ``matplotlib.Figure.set_size_inches`` function.

        Examples
        --------
        >>> p = exp.plot(sample_field='group', gui='jupyter')   # doctest: +SKIP
        >>> p.resize_figure(8, 6)                               # doctest: +SKIP
        >>> p.figure                                            # doctest: +SKIP
        '''
        self.figure.set_size_inches(*args, **kwargs)

    def save_figure(self, *args, **kwargs):
        '''Save the figure to file.

        Parameters
        ----------
        args, kwargs: tuple, dict
            arguments passing to ``matplotlib.Figure.savefig`` function.
        '''
        # try:
        #     # create color bar for the heatmap before saving
        #     self.figure.colorbar(self.ax_hm.images[0])
        # except IndexError:
        #     logger.warning('no heatmap are plotted')
        self.figure.savefig(*args, **kwargs)

    def get_selection_info(self):
        '''Get the current selection information

        Returns
        -------
        tuple of (str, str, numeric)
            sample id, feature id, abundance
        '''
        if self.current_select is None:
            return 'na', 'na', 0
        row, col = self.current_select
        fid = self.exp.feature_metadata.index[col]
        sid = self.exp.sample_metadata.index[row]
        abd = self.exp.data[row, col]
        return sid, fid, abd

    def get_database_annotations(self, feature):
        '''Get database annotations about a feature

        Parameters
        ----------
        feature : str
            The featureID to get info about

        Returns
        -------
        list of tuple of (dict, str)
            dict : annotation key/value pairs
            str : a string summarizing the annotations
        '''
        annt = []
        for cdatabase in self.databases:
            try:
                cannt = cdatabase.get_seq_annotation_strings(feature)
                if len(cannt) == 0:
                    cannt = [[{'annotationtype': 'not found'},
                              'No annotation found in database %s' % cdatabase.database_name]]
                else:
                    for cannotation in cannt:
                        cannotation[0]['_db_interface'] = cdatabase
            except:
                cannt = 'error connecting to db %s' % cdatabase.database_name
            annt.extend(cannt)
        return annt

    def get_info(self):
        '''Get info for the selected feature/sample

        Returns
        -------
        tuple of (str, str, numeric, dict)
            sample id, feature id, abundance, taxonomy, annotation
        '''
        sid, fid, abd = self.get_selection_info()
        annt = self.get_database_annotations(fid)

        return sid, fid, abd, annt

    def show_info(self):
        print(self.get_info())

    def __call__(self):
        '''Run the GUI.'''
        self.connect_functions()
        self.figure.tight_layout()
        # the following line does not work since makes the colorbars far away so commented
        # self.gridspec.tight_layout(self.figure, pad=1, h_pad=0, w_pad=0)

        # squeeze color bars close to the heatmap
        self.figure.subplots_adjust(hspace=0.01, wspace=0.01)

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

        button + super or button + cmd: add current selected features to the selected list
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
        elif event.key == 'super' or event.key == 'cmd':
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
            max_y = self.exp.data.shape[1] - 0.5
            if ylim_lower - y_offset > max_y:
                y_offset = ylim_lower - max_y
            ax.set_ylim(ylim_lower - y_offset, ylim_upper - y_offset)
        elif event.key == 'up':
            if ylim_upper + y_offset < -0.5:
                y_offset = -0.5 - ylim_upper
            ax.set_ylim(ylim_lower + y_offset, ylim_upper + y_offset)
        elif event.key == 'left':
            if xlim_lower - x_offset < -0.5:
                x_offset = xlim_lower + 0.5
            ax.set_xlim(xlim_lower - x_offset, xlim_upper - x_offset)
        elif event.key == 'right':
            max_x = self.exp.data.shape[0] - 0.5
            if xlim_upper + x_offset > max_x:
                x_offset = xlim_upper - max_x
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
            logger.debug('Unrecoginzed key: %s' % event.key)
            return

        self.figure.canvas.draw()

    def clear_selection(self):
        '''Delete all shown selection lines '''
        for cline in self.selected_samples.values():
            self.ax_hm.lines.remove(cline)
            logger.debug('remove sample selection %r' % cline)
        self.selected_samples = {}
        for cline in self.selected_features.values():
            self.ax_hm.lines.remove(cline)
            logger.debug('remove sample selection %r' % cline)
        self.selected_features = {}

    def update_selection(self, samplepos=(), featurepos=(), toggle=True):
        '''Update the selection

        Parameters
        ----------
        samplepos : iterable of int, optional
            positions of samples to be added
        featurepos : iterable of int, optional
            positions of features to be added
        toggle: bool, optional
            True (default) to remove lines in the lists that are already selected.
            False to ignore
        '''
        for cpos in samplepos:
            if cpos not in self.selected_samples:
                self.selected_samples[cpos] = self.ax_hm.axvline(
                    x=cpos, color='white', linestyle='dotted', alpha=0.7, linewidth=0.7)
                logger.debug('add sample selection %r' % cpos)
            else:
                if toggle:
                    self.ax_hm.lines.remove(self.selected_samples[cpos])
                    del self.selected_samples[cpos]
        for cpos in featurepos:
            if cpos not in self.selected_features:
                self.selected_features[cpos] = self.ax_hm.axhline(
                    y=cpos, color='white', linestyle='dotted', alpha=0.7, linewidth=0.7)
                logger.debug('add sample selection %r' % cpos)
            else:
                if toggle:
                    self.ax_hm.lines.remove(self.selected_features[cpos])
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
        return list(self.exp.feature_metadata.index[list(self.selected_features.keys())])
