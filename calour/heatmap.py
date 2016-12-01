# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

from calour.gui.plotgui import class_for_name


logger = getLogger(__name__)


def _transition_index(l):
    '''Return the transition index of the list.

    Parameters
    ----------
    l : list, 1-D array, pd.Series
        l should have method of len and [
    '''
    for i in range(1, len(l)):
        if l[i] != l[i-1]:
            yield i


def plot(exp, xfield=None, feature_field='taxonomy', max_features=40, logit=True, log_cutoff=1, clim=(0, 10), xlabel_rotation=45, cmap=None, title=None, gui='PlotGUI_CLI', axis=None):
    '''Plot an experiment heatmap

    Plot an interactive heatmap for the experiment

    Parameters
    ----------
    xfield : str or None (optional)
        The field to display on the x-axis (sample):
        None (default) to not show x labels.
        str to display field values for this field
    feature_field : str or None (optional)
        Name of the field to display on the y-axis (features) or None not to display names
    max_features : int (optional)
        The maximal number of feature names to display in the plot (when zoomed out)
        0 to show all labels
    logit : bool (optional)
        True (default) to calculate mean of the log2 transformed data (useful for reducing outlier effect)
        False to not log transform before mean calculation
    log_cutoff : float (optional)
        The minimal number of reads for the log trasnform (if logit=True)
    clim : tuple of (float,float) or None (optional)
        the min and max values for the heatmap or None to use all range
    xlabel_rotation : float (optional)
        The rotation angle for the x labels (if xfield is supplied)
    colormap : None or str (optional)
        None (default) to use mpl default color map. str to use colormap named str.
    title : None or str (optional)
        None (default) to show experiment description field as title. str to set title to str.
    gui : str (optional)
        Name of the gui module to use for displaying the heatmap. options:
        'cli' : just cli information about selected sample/feature.
        'qt5' : gui using QT5 (with full bactdb interface)
        'jupyter' : gui for Jupyter notebooks (using widgets)
        Other string : name of child class of plotgui (which should reside in gui/lower(classname).py)
    axis : matplotlib axis or None (optional)
        None (default) to create a new figure, axis to plot heatmap into the axis
    '''
    logger.debug('plot experiment')
    data = exp.get_data(sparse=False, getcopy=True)

    if logit:
        # log transform if needed
        logger.debug('log2 transforming cutoff %f' % log_cutoff)
        data[data < log_cutoff] = log_cutoff
        data = np.log2(data)

    # init the default colormap
    if cmap is None:
        cmap = plt.rcParams['image.cmap']

    # load the appropriate gui module to handle gui events
    if gui == 'qt5':
        gui = 'PlotGUI_QT5'
    elif gui == 'cli':
        gui = 'PlotGUI_CLI'
    elif gui == 'jupyter':
        gui = 'PlotGUI_Jupyter'
    gui_module_name = 'calour.gui.' + gui.lower()
    GUIClass = class_for_name(gui_module_name, gui)
    hdat = GUIClass(exp)

    # init the figure
    if axis is None:
        fig = hdat.get_figure()
        ax = fig.gca()
    else:
        fig = axis.get_figure()
        ax = axis

    hdat.axis = ax
    hdat.fig = fig

    # plot the heatmap
    image = ax.imshow(data.transpose(), aspect='auto', interpolation='nearest', cmap=cmap, clim=clim)

    # plot vertical lines and add x labels for the field
    if xfield is not None:
        if xfield not in exp.sample_metadata:
            raise ValueError('Sample field %s not in sample metadata' % xfield)
        x_values = [exp.sample_metadata[xfield][0]]
        x_pos = [0]
        for transition_pos in _transition_index(exp.sample_metadata[xfield]):
            # samples start -0.5 before and go to 0.5 after
            x_pos.append(transition_pos - 0.5)
            x_values.append(exp.sample_metadata[xfield][transition_pos])
            ax.axvline(x=transition_pos - 0.5, color='white')
        x_pos.append(exp.get_num_samples())
        x_pos = np.array(x_pos)
        ax.set_xticks(x_pos[:-1]+(x_pos[1:]-x_pos[:-1])/2)
        ax.set_xticklabels(x_values, rotation=xlabel_rotation, ha='right')

    # set feature ticks and labels
    if feature_field is not None:
        if feature_field not in exp.feature_metadata:
            raise ValueError('Feature field %s not in feature metadata' % feature_field)
        labels = [x for x in exp.feature_metadata[feature_field]]
        xs = np.arange(len(labels))

        # display only when zoomed enough
        def format_fn(tick_val, tick_pos):
            if int(tick_val) in xs:
                return labels[int(tick_val)]
            else:
                return ''
        if max_features > 0:
            # set the maximal number of feature lables
            ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
            ax.yaxis.set_major_locator(MaxNLocator(max_features, integer=True))
        else:
            # otherwise show all labels
            ax.set_yticks(xs)
            ax.set_yticklabels(labels)
        ax.tick_params(axis='y', which='major', labelsize=8)

    # set the mouse hover string to number of reads
    class Formatter(object):
        def __init__(self, im):
            self.im = im

        def __call__(self, x, y):
            z = self.im.get_array()[int(y), int(x)]
            if logit:
                z = np.power(2, z)
            return 'reads:{:.01f}'.format(z)
    ax.format_coord = Formatter(image)
    # set the title
    if title is None:
        title = exp.description
    else:
        ax.set_title(title)

    try:
        fig.tight_layout()
    except:
        pass

    # link the interactive plot functions
    hdat.connect_functions(fig)

    plt.show()
