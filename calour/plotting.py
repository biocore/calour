'''
plotting (:mod:`calour.plotting`)
=================================

.. currentmodule:: calour.plotting

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   plot_hist
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2017--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import numpy as np
from .util import _transition_index, _to_list
from .heatmap.heatmap import _ax_color_bar


logger = getLogger(__name__)


def plot_hist(exp, ax=None, **kwargs):
    '''Plot histogram of all the values in data.

    It flattens the 2-D array and plots histogram out of it. This
    gives a sense of value distribution. Useful to guess a reasonable
    clim for heatmap.

    Parameters
    ----------
    exp : ``Experiment``
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.
    kwargs : dict
        key word arguments passing to the matplotlib ``hist`` plotting function.

    Returns
    -------
    tuple of 1-D int array, 1-D float array, ``Figure``
        the count in each bin, the start coord of each bin, and hist figure
    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    data = exp.get_data(sparse=False, copy=True)
    counts, bins, patches = ax.hist(data.flatten(), **kwargs)
    # note the count number on top of the histogram bars
    for rect, n in zip(patches, counts):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5,
                int(n), ha='center', va='bottom',
                rotation=90, fontsize=7)
    return counts, bins, fig


def plot_enrichment(exp, enriched, max_show=10, max_len=40, ax=None):
    '''Plot a horizontal bar plot for enriched terms

    Parameters
    ----------
    exp : ``Experiment``
    enriched : pandas.DataFrame
        The enriched terms ( from exp.enrichment() )
        must contain columns 'term', 'odif'
    max_show: int or (int, int) or None (optional)
        The maximal number of terms to show
        if None, show all terms
        if int, show at most the max_show maximal positive and negative terms
        if (int, int), show at most XXX maximal positive and YYY maximal negative terms
    ax: matplotlib Axes or None (optional)
        The axes to which to plot the figure. None (default) to create a new figure

    Returns
    -------
    matplotlib.Figure
        handle to the figure created
    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if max_show is None:
        max_show = [np.inf, np.inf]
    elif isinstance(max_show, int):
        max_show = [max_show, max_show]

    enriched = enriched.sort_values('odif')
    positive = np.min([np.sum(enriched['odif'].values > 0), max_show[0]])
    negative = np.min([np.sum(enriched['odif'].values < 0), max_show[1]])

    ax.barh(np.arange(negative)+positive, enriched['odif'].values[-negative:])
    ax.barh(np.arange(positive), enriched['odif'].values[:positive])
    use = np.zeros(len(enriched), dtype=bool)
    use[:positive] = True
    use[-negative:] = True
    ticks = enriched['term'].values[use]
    ticks = [x.split('(')[0] for x in ticks]
    ticks = ['LOWER IN '+x[1:] if x[0] == '-' else x for x in ticks]
    ticks = [x[:max_len] for x in ticks]
    ax.set_yticks(np.arange(negative+positive), ticks)
    ax.set_xlabel('effect size (positive is higher in group1')
    return fig


def plot_diff_abundance_enrichment(exp, term_type='term', max_show=10, max_len=40, ax=None, ignore_exp=None):
    '''Plot the term enrichment of differentially abundant bacteria

    Parameters
    ----------
    exp : ``Experiment``
        output of differential_abundance()
    max_show: int or (int, int) or None (optional)
        The maximal number of terms to show
        if None, show all terms
        if int, show at most the max_show maximal positive and negative terms
        if (int, int), show at most XXX maximal positive and YYY maximal negative terms
    ax: matplotlib.Axis or None (optional)
        The axis to which to plot the figure
        None (default) to create a new figure
    ignore_exp : list None (optional)
        list of experiment ids to ignore when doing the enrichment_analysis.
        Useful when you don't want to get terms from your own experiment analysis.
        For dbbact it is a list of int
    '''
    if '_calour_diff_abundance_effect' not in exp.feature_metadata.columns:
        raise ValueError('Experiment does not seem to be the results of differential_abundance().')

    # get the positive effect features
    positive = exp.feature_metadata._calour_diff_abundance_effect > 0
    positive = exp.feature_metadata.index.values[positive.values]

    # get the enrichment
    enriched = exp.enrichment(positive, 'dbbact', term_type=term_type, ignore_exp=ignore_exp)

    # and plot
    fig = exp.plot_enrichment(enriched, max_show=max_show, max_len=max_len, ax=ax)
    fig.tight_layout()
    fig.show()
    return fig, enriched


def plot_shareness(exp, group=None, frac_steps=None, ax=None):
    '''Plot the number of shared features against the number of samples included.

    To see if there is a core feature set shared across most of the samples

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax : matplotlib Axes
        The Axes object containing the plot.
    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    else:
        fig = ax.figure



    return fig



def plot_stacked_bar(exp, field=None, sample_color_bars=None, color_bar_label=True, title=None, figsize=(12, 12)):
    '''Plot the number of shared features against the number of samples included.

    To see if there is a core feature set shared across most of the samples

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    fig : matplotlib Figure
        The Figure object containing the plot.
    '''
    from matplotlib.gridspec import GridSpec
    from matplotlib import pyplot as plt

    if exp.sparse:
        data = exp.data.T.toarray()
    else:
        data = exp.data.T
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[12, 1], height_ratios=[3, 12])
    bar = fig.add_subplot(gs[2])

    bottom = np.vstack((np.zeros((data.shape[1],), dtype=data.dtype),
                        np.cumsum(data, axis=0)[:-1]))
    ind = range(data.shape[1])
    rects = []
    for dat, bot in zip(data, bottom):
        rect = bar.barh(ind, dat, left=bot, height=0.9)
        rects.append(rect[0])
    bar.set_yticks(ind)
    bar.set_yticklabels(exp.sample_metadata.index)
    bar.set_ylabel('sample')
    bar.set_xlabel('abundance')
    bar.spines['top'].set_visible(False)
    bar.spines['right'].set_visible(False)
    bar.spines['left'].set_visible(False)

    lax = fig.add_subplot(gs[0])
    lax.axis('off')
    lax.legend(rects, exp.feature_metadata[field], loc="center", labelspacing=0.25, fontsize='small')

    xax = fig.add_subplot(gs[3], sharey=bar)
    xax.axis('off')
    barwidth = 0.3
    barspace = 0.05
    if sample_color_bars is not None:
        sample_color_bars = _to_list(sample_color_bars)
        position = 0
        for s in sample_color_bars:
            # convert to string and leave it as empty if it is None
            values = ['' if i is None else str(i) for i in exp.sample_metadata[s]]
            _ax_color_bar(
                xax, values=values, width=barwidth, position=position, label=color_bar_label, axis=1)
            position += (barspace + barwidth)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    return fig
