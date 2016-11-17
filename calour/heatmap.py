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
import scipy.sparse
import numpy as np

logger = getLogger(__name__)


class HeatmapInteractor(object):
    '''A heatmap editor
    '''
    def __init__(self, fig, exp, zoom_scale=2, scroll_offset=0):
        canvas = fig.canvas
        canvas.mpl_connect('scroll_event', lambda f: scroll_callback(f, hdat=self))
        canvas.mpl_connect('key_press_event', lambda f: key_press_callback(f, hdat=self))
        canvas.mpl_connect('button_press_event', lambda f: button_press_callback(f, hdat=self))
        self.exp = exp
        self.zoom_scale = zoom_scale
        self.scroll_offset = scroll_offset


def button_press_callback(event, hdat):
    rx = int(round(event.xdata))
    ry = int(round(event.ydata))
    print("row: %d    column: %d" % (rx, ry))
    print(hdat.exp.data.shape[0])


def key_press_callback(event, hdat):
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

    if event.key == '[':
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) / hdat.zoom_scale)
    elif event.key == ']':
        ax.set_ylim(
            ylim_lower,
            ylim_lower + (ylim_upper - ylim_lower) * hdat.zoom_scale)
    elif event.key == '{':
        ax.set_xlim(
            xlim_lower,
            xlim_lower + (xlim_upper - xlim_lower) / hdat.zoom_scale)
    elif event.key == '}':
        ax.set_xlim(
            xlim_lower,
            xlim_lower + (xlim_upper - xlim_lower) * hdat.zoom_scale)
    elif event.key == 'down':
        ax.set_ylim(ylim_lower - y_offset, ylim_upper - y_offset)
    elif event.key == 'up':
        ax.set_ylim(ylim_lower + y_offset, ylim_upper + y_offset)
    elif event.key == 'right':
        ax.set_xlim(xlim_lower - x_offset, xlim_upper - x_offset)
    elif event.key == 'left':
        ax.set_xlim(xlim_lower + x_offset, xlim_upper + x_offset)
    else:
        print('Unknown key: %s' % event.key)
        return

    # plt.tight_layout()
    plt.draw()


def scroll_callback(event, hdat):
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

    plt.draw()  # force re-draw


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


def plot(exp, xfield=None, feature_field='taxonomy', max_features=40, logit=True, log_cutoff=1, clim=[0,10], xlabel_rotation=45, cmap=None, title=None):
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
    '''
    logger.debug('plot experiment')
    if scipy.sparse.issparse(exp.data):
        logger.debug('converting from sparse')
        data = exp.data.toarray()
    else:
        data = exp.data.copy()

    if logit:
        # log transform if needed
        logger.debug('log2 transforming cutoff %f' % log_cutoff)
        data[data<log_cutoff] = log_cutoff
        data = np.log2(data)

    # init the default colormap
    if cmap is None:
        cmap = plt.rcParams['image.cmap']

    # plot the heatmap
    fig, ax = plt.subplots()
    ax.imshow(data.transpose(), aspect='auto', interpolation='nearest', cmap=cmap, clim=clim)

    # plot vertical lines and add x labels for the field
    if xfield is not None:
        if xfield not in exp.sample_metadata:
            raise ValueError('Sample field %s not in sample metadata' % xfield)
        x_values=[exp.sample_metadata[xfield][0]]
        x_pos = [0]
        for transition_pos in _transition_index(exp.sample_metadata[xfield]):
            x_pos.append(transition_pos)
            x_values.append(exp.sample_metadata[xfield][transition_pos])
            ax.axvline(x=transition_pos, color='white')
        x_pos.append(exp.data.shape[0])
        x_pos=np.array(x_pos)
        ax.set_xticks(x_pos[:-1]+(x_pos[1:]-x_pos[:-1])/2)
        ax.set_xticklabels(x_values, rotation=xlabel_rotation, ha='center')

    # set feature ticks and labels
    if feature_field is not None:
        if feature_field not in exp.feature_metadata:
            raise ValueError('Feature field %s not in feature metadata' % feature_field)
        labels = [x for x in exp.feature_metadata[feature_field]]
        xs=np.arange(len(labels))

        # display only when zoomed enough
        def format_fn(tick_val, tick_pos):
            if int(tick_val) in xs:
                return labels[int(tick_val)]
            else:
                return ''
        ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
        ax.yaxis.set_major_locator(MaxNLocator(max_features, integer=True))

    # set the title
    if title is None:
        title = exp.description
    if title != '':
        ax.set_title(title)

    # link the interactive plot functions
    HeatmapInteractor(fig, exp)

    plt.show()

