# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import importlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..transforming import log_n


logger = getLogger(__name__)


def _transition_index(l):
    '''Return the transition index and current value of the list.

    Examples
    -------
    >>> l = ['a', 'a', 'b']
    >>> list(_transition_index(l))
    [(2, 'a'), (3, 'b')]

    Parameters
    ----------
    l : Iterable of arbitrary objects

    Yields
    ------
    tuple of (int, arbitrary)
        the transition index, the item value
    '''
    it = enumerate(l)
    i, item = next(it)
    for i, current in it:
        if item != current:
            yield i, item
            item = current
    yield i + 1, item


def create_plot_gui(exp, gui='cli', databases=('dbbact',)):
    '''Create plot GUI object.

    It still waits for the heatmap to be plotted and set up.

    Parameters
    ----------
    gui : str or None (optional)
        If None, just plot a simple matplotlib figure with the heatmap and no interactive elements.
        is str, name of the gui module to use for displaying the heatmap. options:
        'cli' (default) : just cli information about selected sample/feature.
        'qt5' : gui using QT5 (with full dbBact interface)
        'jupyter' : gui for Jupyter notebooks (using widgets)
        Other string : name of child class of plotgui (which should reside in heatmap/lower(classname).py)
    databases : list of str (optional)
        Names of the databases to use to obtain info about sequences. options:
        'dbbact' : the dbBact manual annotation database
        'spongeworld' : the sponge microbiome automatic annotation database
        'redbiom' : the automatic qiita database

    Returns
    -------
    ``PlotGUI`` or its child class
    '''
    # load the gui module to handle gui events & link with annotation databases
    if gui == 'qt5':
        gui = 'PlotGUI_QT5'
    elif gui == 'cli':
        gui = 'PlotGUI_CLI'
    elif gui == 'jupyter':
        gui = 'PlotGUI_Jupyter'
    else:
        raise ValueError('Unknown GUI specified: %r' % gui)
    gui_module_name = 'calour.heatmap.' + gui.lower()
    gui_module = importlib.import_module(gui_module_name)
    GUIClass = getattr(gui_module, gui)
    gui_obj = GUIClass(exp)

    # link gui with the databases requested
    for cdatabase in databases:
        if cdatabase == 'dbbact':
            db_name = 'DBBact'
            db_module_name = 'dbbact_calour.dbbact'
        elif cdatabase == 'spongeworld':
            db_name = 'DBSponge'
            db_module_name = 'dbbact_calour.dbsponge'
        else:
            raise ValueError('Unknown Database specified: %r' % cdatabase)

        # import the database module
        db_module = importlib.import_module(db_module_name)
        # get the class
        DBClass = getattr(db_module, db_name)
        cdb = DBClass()
        gui_obj.databases.append(cdb)
        # select the database for use with the annotate button
        if cdb.annotatable:
            if gui_obj._annotation_db is None:
                gui_obj._annotation_db = cdb
            else:
                logger.warning(
                    'More than one database with annotation capability.'
                    'Using first database (%s) for annotation'
                    '.' % gui_obj._annotation_db.get_name())
    return gui_obj


def heatmap(exp, sample_field=None, feature_field=None, yticklabels_max=100,
            xticklabel_rot=45, xticklabel_len=10, yticklabel_len=15,
            title=None, clim=None, cmap=None,
            axis=None, rect=None,  transform=log_n, **kwargs):
    '''Plot a heatmap for the experiment.

    Plot either a simple or an interactive heatmap for the experiment. Plot features in row
    and samples in column.

    Parameters
    ----------
    sample_field : str or None (optional)
        The field to display on the x-axis (sample):
        None (default) to not show x labels.
        str to display field values for this field
    feature_field : str or None (optional)
        Name of the field to display on the y-axis (features) or None not to display names
    yticklabels_max : int (optional)
        The maximal number of feature names to display in the plot (when zoomed out)
        0 to show all labels
    clim : tuple of (float, float) or None (optional)
        the min and max values for the heatmap or None to use all range. It uses the min
        and max values in the ``data`` array by default.
    xticklabel_rot : float (optional)
        The rotation angle for the x labels (if sample_field is supplied)
    xticklabel_len : int (optional) or None
        The maximal length for the x label strings (will be cut to
        this length if longer). Used to prevent long labels from
        taking too much space. None indicates no cutting
    cmap : None or str (optional)
        None (default) to use mpl default color map. str to use colormap named str.
    title : None or str (optional)
        None (default) to show experiment description field as title. str to set title to str.
    axis : matplotlib ``AxesSubplot`` object or None (optional)
        The axis where the heatmap is plotted. None (default) to create a new figure and
        axis to plot heatmap into the axis
    rect : tuple of (int, int, int, int) or None (optional)
        None (default) to set initial zoom window to the whole experiment.
        [x_min, x_max, y_min, y_max] to set initial zoom window

    Returns
    -------
    ``matplotlib.figure.Figure``

    '''
    logger.debug('plot heatmap')
    numrows, numcols = exp.shape
    # step 1. transform data
    if transform is None:
        data = exp.get_data(sparse=False)
    else:
        logger.debug('transform exp with %r with param %r' % (transform, kwargs))
        data = transform(exp, inplace=False, **kwargs).data

    if axis is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = axis.get_figure(), axis

    # step 2. plot heatmap.
    # init the default colormap
    if cmap is None:
        cmap = plt.rcParams['image.cmap']
    # plot the heatmap
    ax.imshow(data.transpose(), aspect='auto', interpolation='nearest', cmap=cmap, clim=clim)
    # set the title
    if title is None:
        title = exp.description
    ax.set_title(title)
    # set the initial zoom window if supplied
    if rect is not None:
        ax.set_xlim((rect[0], rect[1]))
        ax.set_ylim((rect[2], rect[3]))

    # plot vertical lines between sample groups and add x tick labels
    if sample_field is not None:
        try:
            xticks = _transition_index(exp.sample_metadata[sample_field])
        except KeyError:
            raise ValueError('Sample field %r not in sample metadata' % sample_field)
        ax.set_xlabel(sample_field)
        x_pos, x_val = zip(*xticks)
        x_pos = np.array([0.] + list(x_pos))
        # samples start - 0.5 before and go to 0.5 after
        x_pos -= 0.5
        for pos in x_pos[1:-1]:
            ax.axvline(x=pos, color='white')
        # set tick/label at the middle of each sample group
        ax.set_xticks(x_pos[:-1] + (x_pos[1:] - x_pos[:-1]) / 2)
        xticklabels = [str(i) for i in x_val]
        # shorten x tick labels that are too long:
        if xticklabel_len is not None:
            mid = xticklabel_len / 2
            xticklabels = ['%s..%s' % (i[:mid], i[-mid:])
                           if len(i) > xticklabel_len else i
                           for i in xticklabels]
        ax.set_xticklabels(xticklabels, rotation=xticklabel_rot, ha='right')

    # plot y tick labels dynamically
    if feature_field is not None:
        try:
            ffield = exp.feature_metadata[feature_field]
        except KeyError:
            raise ValueError('Feature field %r not in feature metadata' % feature_field)
        ax.set_ylabel(feature_field)
        yticklabels = [str(i) for i in ffield]
        # for each tick label, show 15 characters at most
        if yticklabel_len is not None:
            yticklabels = [i[-yticklabel_len:] if len(i) > yticklabel_len else i
                           for i in yticklabels]

        def format_fn(tick_val, tick_pos):
            if 0 <= tick_val < numcols:
                return yticklabels[int(tick_val)]
            else:
                return ''
        if yticklabels_max is None:
            # show all labels
            ax.set_yticks(range(numcols))
            ax.set_yticklabels(yticklabels)
        elif yticklabels_max == 0:
            # do not show y labels
            ax.set_yticks([])
        elif yticklabels_max > 0:
            # set the maximal number of feature labels
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_fn))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(yticklabels_max, integer=True))

    # set the mouse hover string to the value of abundance
    def format_coord(x, y):
        row = int(x + 0.5)
        col = int(y + 0.5)
        if 0 <= col < numcols and 0 <= row < numrows:
            z = exp.data[row, col]
            return 'x=%1.2f, y=%1.2f, z=%1.2f' % (x, y, z)
        else:
            return 'x=%1.2f, y=%1.2f' % (x, y)
    ax.format_coord = format_coord

    fig.tight_layout()

    return fig


def plot(exp, gui='cli', databases=('dbbact',), **kwargs):
    gui_obj = create_plot_gui(exp, gui, databases)
    exp.heatmap(axis=gui_obj.axis, **kwargs)
    gui_obj()
    # set up the gui ready for interaction


def plot_sort(exp, field=None, **kwargs):
    '''Plot after sorting by sample field.

    This is a convenience wrapper for plot()

    Note: if sample_field is in **kwargs, use it as labels after sorting using field

    Parameters
    ----------
    field : str or None (optional)
        The field to sort samples by before plotting
    '''
    if field is not None:
        newexp = exp.sort_samples(field)
    else:
        newexp = exp
    if 'sample_field' in kwargs:
        newexp.plot(**kwargs)
    else:
        newexp.plot(sample_field=field, **kwargs)
