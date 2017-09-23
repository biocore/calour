# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import importlib
import itertools

import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np

from ..database import _get_database_class
from .._dendrogram import plot_tree
from ..util import _to_list, _transition_index


logger = getLogger(__name__)


def _create_plot_gui(exp, gui='cli', databases=('dbbact',), tree_size=0):
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
    possible_gui = {'qt5': 'PlotGUI_QT5', 'cli': 'PlotGUI_CLI', 'jupyter': 'PlotGUI_Jupyter'}
    if gui in possible_gui:
        gui = possible_gui[gui]
    else:
        raise ValueError('Unknown GUI specified: %r. Possible values are: %s' % (gui, list(possible_gui.keys())))
    gui_module_name = 'calour.heatmap.' + gui.lower()
    gui_module = importlib.import_module(gui_module_name)
    GUIClass = getattr(gui_module, gui)
    gui_obj = GUIClass(exp, tree_size=tree_size)

    # link gui with the databases requested
    for cdatabase in databases:
        cdb = _get_database_class(cdatabase, exp=exp)
        gui_obj.databases.append(cdb)
        # select the database for use with the annotate button
        if cdb.annotatable:
            if gui_obj._annotation_db is None:
                gui_obj._annotation_db = cdb
            else:
                logger.warning(
                    'More than one database with annotation capability.'
                    'Using first database (%s) for annotation'
                    '.' % gui_obj._annotation_db.database_name)
    return gui_obj


def heatmap(exp, sample_field=None, feature_field=False, yticklabels_max=100,
            xticklabel_rot=45, xticklabel_len=10, yticklabel_len=15,
            title=None, clim=None, cmap=None,
            ax=None, rect=None,  norm=mpl.colors.LogNorm(),
            cax=None, **kwargs):
    '''Plot a heatmap for the experiment.

    Plot either a simple or an interactive heatmap for the experiment. Plot features in row
    and samples in column.

    .. note:: By default it log transforms the abundance values and then plot heatmap.
       The original object is not modified.

    .. _heatmap-ref:

    Parameters
    ----------
    sample_field : str or None (optional)
        The field to display on the x-axis (sample):
        None (default) to not show x labels.
        str to display field values for this field
    feature_field : str or None or False(optional)
        Name of the field to display on the y-axis (features) or None not to display names
        Flase (default) to use the experiment subclass default field
    yticklabels_max : int (optional)
        The maximal number of feature names to display in the plot (when zoomed out)
        0 to show all labels
    clim : tuple of (float, float) or None (optional)
        the min and max values for the heatmap or None to use all range. It uses the min
        and max values in the ``data`` array by default.
    xticklabel_rot : float or None (optional)
        The rotation angle for the x labels (if sample_field is supplied)
        if None, will have rotation=0, horizontalalignment='center', otherwise horizontalalignment='right'
    xticklabel_len : int (optional) or None
        The maximal length for the x label strings (will be cut to
        this length if longer). Used to prevent long labels from
        taking too much space. None indicates no cutting
    cmap : None or str (optional)
        None (default) to use mpl default color map. str to use colormap named str.
    title : None or str (optional)
        None (default) to show experiment description field as title. str to set title to str.
    ax : matplotlib Axes object or None (optional)
        The axes where the heatmap is plotted. None (default) to create a new figure and
        axes to plot heatmap into the axes
    rect : tuple of (int, int, int, int) or None (optional)
        None (default) to set initial zoom window to the whole experiment.
        [x_min, x_max, y_min, y_max] to set initial zoom window
    norm : matplotlib colors Normalize or ``None``
        passed to ``norm`` parameter of ``plt.imshow``. Default is log scale.
    cax : None or matplotlib Axes object (optional)
        plot a legend colorbar for the heatmap in the cax or not

    Returns
    -------
    matplotlib Axes of the heatmap

    '''
    logger.debug('plot heatmap')
    # import pyplot is less polite. do it locally
    import matplotlib.pyplot as plt

    # get the default feature field if not specified (i.e. False)
    if feature_field is False:
        feature_field = exp.heatmap_feature_field
    numrows, numcols = exp.shape
    data = exp.get_data(sparse=False)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # step 2. plot heatmap.
    if title is not None:
        ax.set_title(title)
    # init the default colormap
    if cmap is None:
        cmap = plt.rcParams['image.cmap']
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # this set cells of zero value.
    cmap.set_bad('black')
    # plot the heatmap
    vmin = vmax = None
    if clim is not None:
        vmin, vmax = clim
        # logNorm requires positive values. set it to default None if vmin is zero
        if vmin == 0:
            vmin = None
    image = ax.imshow(data.transpose(), aspect='auto', interpolation='nearest',
                      norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
    # plot legend of color scale
    if cax is not None:
        # make the colorbar wider
        legend = fig.colorbar(image, cax=cax)
        # cax.set_title('percentage', fontdict={'fontsize': 'small'})
        # specify tick label font size
        legend.ax.tick_params(labelsize=9)
        # legend.ax.yaxis.set_major_locator(MaxNLocator(4))
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
        # samples position - 0.5 before and go to 0.5 after
        x_pos -= 0.5
        for pos in x_pos[1:-1]:
            ax.axvline(x=pos, color='white', linewidth=1)
        # set tick/label at the middle of each sample group
        ax.set_xticks(x_pos[:-1] + (x_pos[1:] - x_pos[:-1]) / 2)
        xticklabels = [str(i) for i in x_val]
        # shorten x tick labels that are too long:
        if xticklabel_len is not None:
            mid = int(xticklabel_len / 2)
            xticklabels = ['%s..%s' % (i[:mid], i[-mid:])
                           if len(i) > xticklabel_len else i
                           for i in xticklabels]
        if xticklabel_rot is not None:
            ha = 'right'
        else:
            ha = 'center'
        ax.set_xticklabels(xticklabels, rotation=xticklabel_rot, horizontalalignment=ha)
    else:
        ax.get_xaxis().set_visible(False)

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
    else:
        ax.get_yaxis().set_visible(False)

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
    return ax

def _ax_color_bar(ax, values, width, position=0, colors=None, axis=0, label=True,
                  **label_kwargs):
    '''plot color bars along x or y axis

    Parameters
    ----------
    ax : ``matplotlib`` axes
        the axes to plot the color bars in.
    values : list/tuple
        the values informing the colors on the bar
    width : float
        the width of the color bar
    position : float, optional
        the position of the color bar (its left bottom corner)
    colors : list of colors, optional
        the colors for each unique value in the ``values`` list.
        if it is ``None``, it will use ``Dark2`` discrete color map
        in a cycling way.
    horizontal : bool, optional
        plot the color bar horizontally or vertically
    label : bool, optional
        whether to label the color bars with text
    **label_kwargs: dict
        keyword arguments to pass in for annotating labels

    Returns
    -------
    ``matplotlib`` axes
    '''

    default_kwargs = {'color': 'w', 'weight': 'bold', 'fontsize': 7,
                      'ha': 'center', 'va':'center'}
    for k, v in default_kwargs.items():
        if k not in label_kwargs.keys():
            label_kwargs[k] =v

    uniques = np.unique(values)
    if colors is None:
        cmap = mpl.cm.get_cmap('Dark2')
        colors = cmap.colors
    col = dict(zip(uniques, itertools.cycle(colors)))
    prev = 0
    offset = 0.5
    for i, value in _transition_index(values):
        if value != '':
            # do not plot the current segment of the bar
            # if the value is empty
            if axis == 0:
                # plot the color bar along x axis
                pos = prev - offset, position
                w, h = i - prev, width
                rotation = 0
            else:
                # plot the color bar along y axis
                pos = position, prev - offset
                w, h = width, i - prev
                rotation = 90
            rect = mpatches.Rectangle(
                pos,               # position
                w,                 # width (size along x axis)
                h,                 # height (size along y axis)
                edgecolor="none",  # No border
                facecolor=col[value],
                label=value)
            ax.add_patch(rect)
            if label is True:
                rx, ry = rect.get_xy()
                cx = rx + rect.get_width() / 2.0
                cy = ry + rect.get_height() / 2.0

                # add the text in the color bars
                ax.annotate(value, (cx, cy), rotation=rotation,
                            **label_kwargs)

        prev = i

    return ax


def plot(exp, sample_color_bars=None, feature_color_bars=None,
         gui='cli', databases=False, color_bar_label=True,
         tree=None, tree_size=8, title=None, barwidth = 0.3,
         barspace = 0.05, label_kwargs={}, **kwargs):

    '''Plot the interactive heatmap and its associated axes.

    The heatmap is interactive and can be dynamically updated with
    following key and mouse events:

    +---------------------------+-----------------------------------+
    |Event                      |Description                        |
    +===========================+===================================+
    |`+` or `⇧ →`               |zoom in on x axis                  |
    |                           |                                   |
    +---------------------------+-----------------------------------+
    |`_` or `⇧ ←`               |zoom out on x axis                 |
    |                           |                                   |
    +---------------------------+-----------------------------------+
    |`=` or `⇧ ↑`               |zoom in on y axis                  |
    |                           |                                   |
    +---------------------------+-----------------------------------+
    |`-` or `⇧ ↓`               |zoom out on y axis                 |
    |                           |                                   |
    +---------------------------+-----------------------------------+
    |`left mouse click`         |select the current row and column  |
    +---------------------------+-----------------------------------+
    |`⇧` and `left mouse click` |select all the rows between        |
    |                           |previous selected and current rows |
    +---------------------------+-----------------------------------+
    |`.`                        |move the selection down by one row |
    +---------------------------+-----------------------------------+
    |`,`                        |move the selection up by one row   |
    +---------------------------+-----------------------------------+
    |`<`                        |move the selection left by one     |
    |                           |column                             |
    +---------------------------+-----------------------------------+
    |`>`                        |move the selection right by one    |
    |                           |column                             |
    +---------------------------+-----------------------------------+
    |`↑` or `=`                 |scroll the heatmap up on y axis    |
    +---------------------------+-----------------------------------+
    |`↓` or `-`                 |scroll the heatmap down on y axis  |
    +---------------------------+-----------------------------------+
    |`←` or `<`                 |scroll the heatmap left on x axis  |
    +---------------------------+-----------------------------------+
    |`→` or `>`                 |scroll the heatmap right on x axis |
    +---------------------------+-----------------------------------+


    .. _plot-ref:

    Parameters
    ----------
    exp : ``Experiment``
        the object to plot
    sample_color_bars : list or str, optional
        list of column names in the sample metadata. It plots a color bar
        for each column. It doesn't plot color bars by default (``None``)
    feature_color_bars : list or str, optional
        list of column names in the feature metadata. It plots a color bar
        for each column. It doesn't plot color bars by default (``None``)
    color_bar_label : bool, optional
        whether to show the label for the color bars
    gui : str, optional
        GUI to use
    databases : Iterable of str or None or False (optional)
        a list of databases to access or add annotation
        False (default) to use the default field based on the experiment subclass
        None to not use databases
    tree : skbio.TreeNode or None (optional)
        None (default) to not plot a tree
        otherwise, plot the tree dendrogram on the left.
        NOTE: features are reordered according to the tree
    tree_size : int (optional)
        The width of the tree relative to the main heatmap (12 is identical size)
    title : str (optional)
        The title of the figure.
    barwidth : float
        The width of the bars
    barspace : float
        The spacing between the bars.
    **label_kwargs : dict, optional
        keyword arguments passing to :ref:`_ax_color_bar` function
        to modify the labels.
    **kwargs : dict, optional
        keyword arguments passing to :ref:`heatmap<heatmap-ref>` function.

    Returns
    -------
    ``PlottingGUI``
        Contains the figure of the output plot in .figure parameter
    '''
    # set the databases if default requested (i.e. False)
    if databases is False:
        databases = exp.heatmap_databases

    if tree is None:
        gui_obj = _create_plot_gui(exp, gui, databases)
    else:
        gui_obj = _create_plot_gui(exp, gui, databases, tree_size=tree_size)
        # match the exp order to the tree (reorders the features)
        exp, tree = plot_tree(exp, tree, gui_obj.tree_axes)

    if title is not None:
        gui_obj.figure.suptitle(title)

    exp.heatmap(ax=gui_obj.axes, cax=gui_obj.legend, **kwargs)

    if sample_color_bars is not None:
        sample_color_bars = _to_list(sample_color_bars)
        position = 0
        for s in sample_color_bars:
            # convert to string and leave it as empty if it is None
            values = ['' if i is None else str(i) for i in exp.sample_metadata[s]]
            _ax_color_bar(
                gui_obj.xax, values=values, width=barwidth, position=position,
                label=color_bar_label, axis=0,
                **label_kwargs)

            position += (barspace + barwidth)
    if feature_color_bars is not None:
        feature_color_bars = _to_list(feature_color_bars)
        position = 0
        for f in feature_color_bars:
            values = ['' if i is None else str(i) for i in exp.feature_metadata[f]]
            _ax_color_bar(
                gui_obj.yax, values=values, width=barwidth, position=position, label=color_bar_label, axis=1,
                **label_kwargs)
            position += (barspace + barwidth)
    # set up the gui ready for interaction
    gui_obj()

    return gui_obj


def plot_sort(exp, fields=None, sample_color_bars=None, feature_color_bars=None,
              gui='cli', databases=False, color_bar_label=True, title=None, **kwargs):
    '''Plot after sorting by sample field.

    This is a convenience wrapper for plot().

    .. note:: Sorting occurs on a copy, the original ``Experiment`` object is not modified.

    Parameters
    ----------
    fields : str, list, or None, optional
        The fields to sort samples by before plotting
    sample_color_bars : list, optional
        list of column names in the sample metadata. It plots a color bar
        for each column. It doesn't plot color bars by default (``None``)
    feature_color_bars : list, optional
        list of column names in the feature metadata. It plots a color bar
        for each column. It doesn't plot color bars by default (``None``)
    color_bar_label : bool, optional
        whether to show the label for the color bars
    gui : str, optional
        GUI to use:
        'cli' : simple command line gui
        'jupyter' : jupyter notebook interactive gui
        'qt5' : qt5 based interactive gui
        None : no interactivity - just a matplotlib figure
    databases : Iterable of str or None or False (optional)
        a list of databases to access or add annotation
        False (default) to use the default field based on the experiment subclass
        None to not use databases
    title : str (optional)
        The title of the figure.
    kwargs : dict, optional
        keyword arguments passing to :ref:`plot<plot-ref>` function.

    Returns
    -------
    PlotGUI
    '''
    if fields is not None:
        newexp = exp.copy()
        fields = _to_list(fields)
        for cfield in fields:
            newexp.sort_samples(cfield, inplace=True)
        plot_field = cfield
    else:
        newexp = exp
        plot_field = None
    if 'sample_field' in kwargs:
        return newexp.plot(sample_color_bars=sample_color_bars, feature_color_bars=feature_color_bars,
                           gui=gui, databases=databases, color_bar_label=color_bar_label, title=title, **kwargs)
    else:
        return newexp.plot(sample_field=plot_field, sample_color_bars=sample_color_bars, feature_color_bars=feature_color_bars,
                           gui=gui, databases=databases, color_bar_label=color_bar_label, title=title, **kwargs)
