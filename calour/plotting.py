'''
plotting (:mod:`calour.plotting`)
=================================

.. currentmodule:: calour.plotting

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   plot_hist
   plot_enrichment
   plot_diff_abundance_enrichment
   plot_stacked_bar
   plot_shareness
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
from .util import _to_list, compute_prevalence
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


def plot_shareness(exp, field=None, step=3, steps=None, iterations=10, ax=None):
    '''Plot the number of shared features against the number of samples subsampled.

    To see if there is a core feature set shared across most of the samples.

    As an example of this type of plot, please see Fig 2C in Muegge,
    B. D. et al. Diet drives convergence in gut microbiome functions
    across mammalian phylogeny and within humans. Science 332, 970–974
    (2011).


    Parameters
    ----------
    field : str
        sample metadata field to group samples
    step : int
        step size to compute the shareness
    steps : iterable of ints
        steps to compute the shareness. If it is specified, it overrides ``step``.
    iterations : int
        repeat the compute multiple times and plot all the iterations
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

    if field is None:
        for i in range(iterations):
            x, y = _compute_frac_nonzero(exp.data, step, steps)
            if i == 0:
                line, = ax.plot(x, y * 100, alpha=0.5)
            else:
                # use the same color as the first iteration
                ax.plot(x, y * 100, alpha=0.5, color=line.get_color())
    else:
        for uniq in exp.sample_metadata[field].unique():
            data = exp.filter_samples(field, uniq).data
            for i in range(iterations):
                x, y = _compute_frac_nonzero(data, step, steps)
                if i == 0:
                    line, = ax.plot(x, y * 100, label=uniq, alpha=0.5)
                else:
                    ax.plot(x, y * 100, alpha=0.5, color=line.get_color())
        ax.legend()
    ax.set_xlabel('sample number')
    ax.set_ylabel('shared features (%)')
    return ax


def _compute_frac_nonzero(data, step, steps):
    '''iteratively compute the fraction of non-zeros in each column after subsampling rows. '''
    n, features = data.shape
    if steps is None:
        steps = [i for i in range(2, n, step)][::-1]
    else:
        # filter out the illegal large values
        steps = sorted([i for i in steps if i < n], reverse=True)
    shared = []
    for i in steps:
        data = data[np.random.choice(n, int(i), replace=False), :]
        x = data > 0
        # the count of samples that have the given feature
        counts = x.sum(axis=0).A1
        frac = np.sum(counts == i)
        shared.append(frac)
        n = data.shape[0]
    shared = np.array(shared) / features
    return steps, shared


def plot_abund_prevalence(exp, field, log=True, min_abund=0.01, ax=None):
    '''Plot abundance against prevalence.

    Prevalence/abundance curve is a chart used to visualize the
    prevalence of OTUs. For each OTU, a curve was constructed
    measuring the percentage of a population that carries the OTU
    above a given abundance (normalized over the total abundance of
    the OTU). A steep curve indicates this OTU is shared prevalently
    among the population. If many OTUs show in steep curves, it
    indicates the population has a core set of microbes.

    Y-axis: prevalence of the OTU that above the abundance threshold.

    X-axis: abundance threshold.

    As an example of this type of plot, please see Fig 1D in Clemente,
    J. C. et al. The microbiome of uncontacted Amerindians. Science
    Advances 1, e1500183 (2015).

    .. warning:: This function is still less tested.

    Parameters
    ----------
    field : str
        sample metadata field to group samples
    log : bool
        whether to plot abundance in log scale
    min_abund : numeric
        the min abundance. features with mean abundance
        less than min_abund in the each sample group will be not considered
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()

    for uniq in exp.sample_metadata[field].unique():
        data = exp.filter_samples(
            field, uniq).filter_by_data(
                'mean_abundance', cutoff=min_abund, axis=1).data
        flag = True
        # this implementation is for both dense and sparse arrays
        for column in range(data.shape[1]):
            feature = data[:, column].data
            x, y = compute_prevalence(feature)
            if flag:
                line, = ax.plot(x, y, alpha=0.5, label=uniq)
                flag = False
            else:
                ax.plot(x, y, alpha=0.5, color=line.get_color())

    ax.set_ylabel('prevalence')
    if log is True:
        ax.set_xscale("log", nonposx='mask')
        ax.set_xlabel('log(abundance)')
    else:
        ax.set_xlabel('abundance')
    # ax.invert_xaxis()
    ax.legend()
    return ax


def plot_stacked_bar(exp, sample_color_bars=None, color_bar_label=True, title=None,
                     figsize=(12, 8), legend_size='small', legend_field=None, xtick=False):
    '''Plot the stacked bar for feature abundances.

    Parameters
    ----------
    xtick : str, False, or None
        how to draw ticks and tick labels on x axis.
        str: use a column name in sample metadata;
        None: use sample IDs;
        False: do not draw ticks.
    sample_color_bars : list, optional
        list of column names in the sample metadata. It plots a color bar
        for each unique column to indicate sample group. It doesn't plot color bars by default (``None``)
    color_bar_label : bool
        whether to show the label on the color bars
    title : str
        figure title
    figsize : tuple of numeric
        figure size passed to ``figsize`` in ``plt.figure``
    legend_size : str or int
        passed to ``fontsize`` in ``ax.legend()``
    legend_field : str, or None
        a column name in feature metadata. the values in the column will be used as the legend labels

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
    gs = GridSpec(2, 2, width_ratios=[12, 6], height_ratios=[1, 12])

    bar = fig.add_subplot(gs[2])
    bottom = np.vstack((np.zeros((data.shape[1],), dtype=data.dtype),
                        np.cumsum(data, axis=0)[:-1]))
    ind = range(data.shape[1])
    rects = []
    for dat, bot in zip(data, bottom):
        rect = bar.bar(ind, dat, bottom=bot, width=0.95)
        rects.append(rect[0])
    if xtick is None:
        bar.set_xticks(ind)
        bar.set_xticklabels(exp.sample_metadata.index, rotation='vertical')
    elif xtick is False:
        # don't draw tick and tick label on x axis
        bar.tick_params(labelbottom='off', bottom='off')
    else:
        bar.set_xticks(ind)
        bar.set_xticklabels(exp.sample_metadata[xtick], rotation='vertical')

    bar.set_xlabel('sample')
    bar.set_ylabel('abundance')
    bar.spines['top'].set_visible(False)
    bar.spines['right'].set_visible(False)
    bar.spines['bottom'].set_visible(False)

    xax = fig.add_subplot(gs[0], sharex=bar)
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
                xax, values=values, width=barwidth, position=position, label=color_bar_label, axis=0)
            position += (barspace + barwidth)

    if legend_field is not None:
        lax = fig.add_subplot(gs[3])
        lax.axis('off')
        lax.legend(rects, exp.feature_metadata[legend_field], loc="center left", fontsize=legend_size)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    return fig
