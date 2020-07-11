'''
other plotting functions (:mod:`calour.plotting`)
=================================================

.. currentmodule:: calour.plotting

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   plot_hist
   plot_enrichment
   plot_diff_abundance_enrichment
   plot_stacked_bar
   plot_core_features
   plot_abund_prevalence
   plot_feature_matrix
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2017--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from itertools import cycle, combinations

import numpy as np
from scipy import stats

from . import Experiment
from .util import _to_list, compute_prevalence
from .heatmap.heatmap import _ax_bar
from .analysis import _CALOUR_STAT, _CALOUR_DIRECTION


logger = getLogger(__name__)


def plot_hist(exp: Experiment, ax=None, **kwargs):
    '''Plot histogram of all the values in data.

    It flattens the 2-D array and plots histogram out of it. This
    gives a sense of value distribution. Useful to guess a reasonable
    clim for heatmap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None (default), optional
        Axes object to draw the plot onto; otherwise uses the current Axes.
    kwargs : dict
        key word arguments passing to the :func:`matplotlib.pyplot.hist` plotting function.

    Returns
    -------
    tuple of 1-D int array, 1-D float array, matplotlib.axes.Axes
        the count in each bin, the start coord of each bin, and hist figure
    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()

    data = exp.get_data(sparse=False, copy=True)
    counts, bins, patches = ax.hist(data.flatten(), **kwargs)
    # note the count number on top of the histogram bars
    for rect, n in zip(patches, counts):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5,
                int(n), ha='center', va='bottom',
                rotation=90, fontsize=7)
    return counts, bins, ax


def plot_enrichment(exp: Experiment, enriched, max_show=10, max_len=40, ax=None, labels=('group1', 'group2'), colors=('green', 'red'), enriched_exp_color='white'):
    '''Plot a horizontal bar plot for enriched terms

    Parameters
    ----------
    enriched : pandas.DataFrame
        The enriched terms ( from exp.enrichment(). i.e. if res=exp.enrichment, use res[0] )
        must contain columns 'term', 'odif'
    max_show: int or (int, int) or None, optional
        The maximal number of terms to show
        if None, show all terms
        if int, show at most the max_show maximal positive and negative terms
        if (int, int), show at most XXX maximal positive and YYY maximal negative terms
    ax: matplotlib.axes.Axes or None, optional
        The axes to which to plot the figure. None (default) to create a new figure
    lables: tuple of (str, str) or None (optional)
        name for terms enriched in group1 or group2 respectively, or None to not show legend
    colors: tuple of (str, str) or None (optional)
        Colors for terms enriched in group1 or group2 respectively
    enriched_exp_color: str or None, optional
        If not None, the color to show the number of enriched experiments for each term in the bar. Default is white since the background is the bar color (green/red).
        None to not show the enriched experiments count


    Returns
    -------
    matplotlib.axes.Axes
    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()

    if max_show is None:
        max_show = [np.inf, np.inf]
    elif isinstance(max_show, int):
        max_show = [max_show, max_show]

    enriched = enriched.sort_values('odif')
    evals = enriched['odif'].values
    positive = np.min([np.sum(enriched['odif'].values > 0), max_show[0]])
    negative = np.min([np.sum(enriched['odif'].values < 0), max_show[1]])
    if negative + positive == 0:
        logger.warning('No significantly enriched categories found')
        return ax
    if positive > 0:
        ax.barh(np.arange(positive) + negative, evals[-positive:], color=colors[0])
    if negative > 0:
        ax.barh(np.arange(negative), evals[:negative], color=colors[1])
    use = np.zeros(len(enriched), dtype=bool)
    if positive > 0:
        use[-positive:] = True
    if negative > 0:
        use[:negative] = True

    ticks = enriched['term'].values[use]
    ticks = [x[:max_len] for x in ticks]
    ax.set_yticks(np.arange(negative + positive))
    ax.set_yticklabels(ticks, style='italic', weight='semibold')
    if labels is not None:
        ax.set_xlabel('effect size (positive is higher in %s)' % labels[0])
        ax.legend(labels)

    # add the number of enriched experiments for each term
    if enriched_exp_color is not None and 'num_enriched_exps' in enriched:
        if positive > 0:
            for idx in np.arange(negative, negative + positive):
                cnum_enriched = enriched.iloc[idx]['num_enriched_exps']
                if cnum_enriched < 0:
                    continue
                cstr = ' %d/%d' % (cnum_enriched, enriched.iloc[idx]['num_total_exps'])
                ax.text(0, idx, cstr, horizontalalignment='left', verticalalignment='center', color=enriched_exp_color, fontweight='bold')
        if negative > 0:
            for idx in np.arange(negative):
                cnum_enriched = enriched.iloc[idx]['num_enriched_exps']
                if cnum_enriched < 0:
                    continue
                cstr = '%d/%d ' % (cnum_enriched, enriched.iloc[idx]['num_total_exps'])
                ax.text(0, idx, cstr, horizontalalignment='right', verticalalignment='center', color=enriched_exp_color, fontweight='bold')

    ax.figure.tight_layout()
    return ax


def plot_diff_abundance_enrichment(exp: Experiment, max_show=10, max_len=40, ax=None, colors=('green', 'red'), show_legend=True, enriched_exp_color='white', **kwargs):
    '''Plot the term enrichment of differentially abundant bacteria

    Parameters
    ----------
    term_type : str (optional)
        What types of annotations/terms to include in enrichment analysis.
        Options are:
        'term' - ontology terms associated with each feature.
        'parentterm' - ontology terms including parent terms associated with each feature.
        'annotation' - the full annotation strings associated with each feature
        'combined' - combine 'term' and 'annotation'
    max_show: int or (int, int) or None (optional)
        The maximal number of terms to show
        if None, show all terms
        if int, show at most the max_show maximal positive and negative terms
        if (int, int), show at most XXX maximal positive and YYY maximal negative terms
    ax: matplotlib.axes.Axes or None, optional
        The axis to which to plot the figure
        None (default) to create a new figure
    colors: tuple of (str, str) or None (optional)
        Colors for terms enriched in group1 or group2 respectively
    show_legend: bool (optional)
        True to show the color legend, False to hide it
    enriched_exp_color: str or None, optional
        If not None, the color to show the number of enriched
        experiments for each term in the bar. Default is white since
        the background is the bar color (green/red).  None to not show
        the enriched experiments count
    **kwargs : dict, optional
        Additional database specific enrichment parameters (see per-database module documentation for .enrichment() method)

    Returns
    -------
    tuple of :class:`matplotlib.axes.Axes`, :class:`.Experiment` with
    terms as features, (original features) as samples, result of
    :func:`.database.enrichment`

    '''
    if _CALOUR_DIRECTION not in exp.feature_metadata.columns:
        raise ValueError('Experiment does not seem to be the result of differ_abundance().')

    # get the positive effect features
    positive = exp.feature_metadata[_CALOUR_STAT] > 0
    positive = exp.feature_metadata.index.values[positive.values]

    # get the enrichment
    enriched, term_features, features = exp.enrichment(features=positive, dbname='dbbact', **kwargs)
    # features=pd.DataFrame({'sequence': features}, index=features)

    # Create an new experiment where features are the enriched terms, and samples are the features
    # The newexp.feature_metadata contains the 'odif', 'pval' fields for each term
    newexp = Experiment(term_features, sample_metadata=features, feature_metadata=enriched)

    # get the labels for the two groups
    if show_legend:
        labels = ['group1', 'group2']
        names1 = exp.feature_metadata[_CALOUR_DIRECTION][exp.feature_metadata[_CALOUR_STAT] > 0]
        if len(names1) > 0:
            labels[0] = names1.values[0]
        names2 = exp.feature_metadata[_CALOUR_DIRECTION][exp.feature_metadata[_CALOUR_STAT] < 0]
        if len(names2) > 0:
            labels[1] = names2.values[0]
    else:
        labels = None

    # and plot
    ax2 = exp.plot_enrichment(enriched, max_show=max_show, max_len=max_len, ax=ax, labels=labels, colors=colors, enriched_exp_color=enriched_exp_color)

    return ax2, newexp


def plot_core_features(exp: Experiment, field=None, steps=None, cutoff=2, frac=0.9, iterations=10, alpha=0.5, linewidth=0.7, ax=None):
    '''Plot the percentage of core features shared in increasing number of samples.

    To see if there is a core feature set shared across most of the samples.

    As an example of this type of plot, please see Fig 2C in Muegge,
    B. D. et al. Diet drives convergence in gut microbiome functions
    across mammalian phylogeny and within humans. Science 332, 970–974
    (2011).

    .. warning:: The samples should be normalized by rarefaction
       instead of other normalization methods. Otherwise, the samples
       with higher sequencing depth will artificially share more OTUs
       with each other.

    Parameters
    ----------
    field : str
        sample metadata field to group samples
    steps : iterable of int
        the sizes of subsamples to compute the fraction of core features.
    cutoff : numeric
        the feature is considered present in a sample only if its abundance is >= cutoff.
    frac : numeric
        Must between 0 and 1. The feature would be considered as a core feature
        if it is present in ``fac`` faction of samples.
    iterations : int
        repeat the subsampling multiple times and plot all the iterations
    alpha : float
        the transparency (1 is most opaque and 0 is most transparent) to plot for each iteration.
    linewidth : float
        the linewidth of the plotting lines for each iteration
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()

    def plot_lines(data, steps, label):
        y_sum = np.zeros(len(steps))
        for i in range(iterations):
            y = _compute_frac_nonzero(data, steps, cutoff, frac, i)
            y_sum += y
            if i == 0:
                line, = ax.plot(steps, y, alpha=alpha, linewidth=linewidth)
            else:
                ax.plot(steps, y, alpha=alpha, linewidth=linewidth, color=line.get_color())
        y_ave = y_sum / iterations
        # plot average of the iterations
        ax.plot(steps, y_ave, linewidth=linewidth * 3, label=label, color=line.get_color())

    if field is None:
        plot_lines(exp.data, steps, label='all samples')
    else:
        for uniq in exp.sample_metadata[field].unique():
            data = exp.filter_samples(field, uniq).data
            # this subset may have number of samples smaller than some value in steps
            steps_i = [i for i in steps if i <= data.shape[0]]
            plot_lines(data, steps_i, uniq)
        ax.legend()
    # because the shareness drops quickly, we plot it in log scale
    ax.set_xscale('log')
    ax.set_xlabel('sample number')
    ax.set_ylabel('shared features (%)')
    return ax


def _compute_frac_nonzero(data, steps=None, cutoff=2, frac=0.9, random_seed=None):
    '''iteratively compute the fraction of non-zeros in each column after subsampling rows.

    Parameters
    ----------
    data : 2-d array of numeric
        sample in row and feature in column
    steps : iterable of int
        the subsample sizes (should be in descending order)
    cutoff : numeric
        the feature is considered present in a sample only if its abundance is >= cutoff.
    frac : numeric
        Must between 0 and 1. The feature would be considered as a core feature
        if it is present in ``fac`` faction of samples.
    random_seed : int, np.radnom.Generator instance or None, optional, default=None
        set the random number generator seed
        If int, random_seed is the seed used by the random number generator;
        If Generator instance, random_seed is set to the random number generator;
        If None, then fresh, unpredictable entropy will be pulled from the OS

    Return
    ------
    numpy.array
        fractions of core features for each subsample size
    '''
    n_samples, n_features = data.shape

    # filter out the illegal large values
    if steps is None:
        steps = [int(i) for i in np.logspace(2, np.log2(n_samples), num=7)]
    else:
        steps = sorted([i for i in steps if i <= n_samples], reverse=True)
    logger.debug('steps are filtered and sorted to %r' % steps)

    shared = np.zeros(len(steps))
    rng = np.random.default_rng(random_seed)
    if cutoff <= 0:
        raise ValueError('You need to provide a positive value for `cutoff`: %r' % cutoff)
    if frac <= 0 or frac > 1:
        raise ValueError('You need to provide a value among (0, 1] for `frac`: %r' % frac)
    for n, i in enumerate(steps):
        data = data[rng.choice(n_samples, i, replace=False), :]
        x = data >= cutoff
        # the count of samples that have the given feature
        counts = x.sum(axis=0)
        all_presence = np.sum(counts >= np.ceil(i * frac))
        all_absence = np.sum(counts == 0)
        # important: remove the features that are all zeros across the subset of samples
        shared[n] = all_presence / (n_features - all_absence)
        # don't forget to update sample count
        n_samples = data.shape[0]
    return shared


def plot_abund_prevalence(exp: Experiment, field, log=True, min_abund=0.01, alpha=0.5, linewidth=0.7, ax=None):
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
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.

    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()

    for uniq in exp.sample_metadata[field].unique():
        data = exp.filter_samples(field, uniq).filter_by_data('abundance', cutoff=min_abund, axis=1, mean_or_sum='mean').data
        flag = True
        # this implementation is for both dense and sparse arrays
        for column in range(data.shape[1]):
            feature = data[:, column].data
            x, y = compute_prevalence(feature)
            if flag:
                line, = ax.plot(x, y, label=uniq, alpha=alpha, linewidth=linewidth)
                flag = False
            else:
                ax.plot(x, y, color=line.get_color(), alpha=alpha, linewidth=linewidth)

    ax.set_ylabel('prevalence')
    if log is True:
        ax.set_xscale("log", nonposx='mask')
        ax.set_xlabel('log(abundance)')
    else:
        ax.set_xlabel('abundance')
    # ax.invert_xaxis()
    ax.legend()
    return ax


def plot_stacked_bar(exp: Experiment, field=None, sample_color_bars=None,
                     color_bar_label=True,
                     barx_label_kwargs=None, barx_width=0.3, barx_colors=None,
                     title=None, figsize=(12, 8), legend_size='small',
                     xtick=False, cmap='Paired'):
    '''Plot the stacked bar for feature abundances.

    Parameters
    ----------
    field : str, or None
        a column name in feature metadata. the values in the column will be used as the legend labels
    sample_color_bars : list, optional
        list of column names in the sample metadata. It plots a color bar
        for each unique column to indicate sample group. It doesn't plot color bars by default (``None``)
    color_bar_label : bool
        whether to show the label on the color bars
    barx_label_kwargs : dict, optional
        keyword arguments to pass in for :func:`matplotlib.axes.Axes.annotate`
        for labels on the X-axis bar
    barx_width : float, optional
        the width of the color bar
    barx_colors : dict, optional
        the colors for each unique value in the ``values`` list.
        if it is ``None``, it will use ``Dark2`` discrete color map
        in a cycling way.
    title : str
        figure title
    figsize : tuple of numeric
        figure size passed to ``figsize`` in ``plt.figure``
    legend_size : str or int
        passed to ``fontsize`` in ``ax.legend()``
    xtick : str, False, or None
        how to draw ticks and tick labels on x axis.
        str: use a column name in sample metadata;
        None: use sample IDs;
        False: do not draw ticks.
    cmap : string
        matplotlib qualitative colormap

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot.
    '''
    from matplotlib.gridspec import GridSpec
    from matplotlib import pyplot as plt

    if exp.sparse:
        data = exp.data.T.toarray()
    else:
        data = exp.data.T

    cmap = plt.get_cmap(cmap)
    colors = cycle(cmap.colors)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[12, 6], height_ratios=[1, 12])

    bar = fig.add_subplot(gs[2])
    bottom = np.vstack((np.zeros((data.shape[1],), dtype=data.dtype),
                        np.cumsum(data, axis=0)[:-1]))
    ind = range(data.shape[1])
    rects = []
    for dat, bot, col in zip(data, bottom, colors):
        rect = bar.bar(ind, dat, bottom=bot, width=0.95, color=col)
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
    barwidth = barx_width
    barspace = 0.05
    if sample_color_bars is not None:
        sample_color_bars = _to_list(sample_color_bars)
        position = 0
        for s in sample_color_bars:
            # convert to string and leave it as empty if it is None
            values = ['' if i is None else str(i) for i in exp.sample_metadata[s]]
            _ax_bar(xax, values=values, width=barwidth, position=position,
                    colors=barx_colors, label_kwargs=barx_label_kwargs,
                    label=color_bar_label, axis=0)
            position += (barspace + barwidth)

    if field is not None:
        lax = fig.add_subplot(gs[3])
        lax.axis('off')
        lax.legend(rects, exp.feature_metadata[field], loc="center left", fontsize=legend_size)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    return fig


def plot_feature_matrix(exp: Experiment, fields, feature_ids, title_field=None,
                        transform_x=None, transform_y=None, plot='scatter',
                        ncols=5, nrows=None, size=2, aspect=1, **kwargs):
    '''This plots an array of scatter plots between each features against the specified sample metadata.

    For each panel of scatter plot, the x-axis is the co-variates
    specified by sample metadata field; the y-axis is the feature
    abundance.

    Parameters
    ----------
    fields : str
        the column in the sample metadata to plot against
    feature_ids : list-like
        the IDs of features
    transform_x, transform_y : callable
        the transformation for values on x- and y-axis
    plot : str, {'scatter', 'box'}
        the plot type
    ncols, nrows : int
        plot nrows x ncols number of scatter plots. If nrows is None, then its value is determined
        automatically by the-number-features / ncols
    size : numeric
        the height of each figure panel in inches
    aspect : numeric
        Aspect ratio of each figure panel, so that aspect * size gives its width
    kwargs : dict
        keyword arguments passing to either :func:`matplotlib.pyplot.boxplot` or
        :func:`matplotlib.pyplot.scatter`, depending on `plot` argument.

    Returns
    -------
    matplotlib.figure.Figure
    '''
    from matplotlib import pyplot as plt

    x = exp.sample_metadata[fields].values

    if plot == 'scatter':
        if transform_x is not None:
            x = transform_x(x)
    d = {'box': plot_box, 'scatter': plot_scatter}
    plot = d[plot]

    if nrows is None:
        # get the ceiling of the division
        nrows = -(-len(feature_ids) // ncols)
        max_rows = 50
        if nrows > max_rows:
            raise ResourceWarning('There are over %d figure rows; it will be slow. '
                                  'If you really want to plot this number of figures, '
                                  'please explicitly specify it in the nrows function argument.' % max_rows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(size * ncols * aspect, size * nrows))

    for ax, fid in zip(axs.flat, feature_ids):
        # y is 1d np array
        y = exp[:, fid]
        if transform_y is not None:
            y = transform_y(y)
        if title_field is None:
            title = ''
        else:
            title = exp.feature_metadata.loc[fid, title_field]
        plot(x, y, title, ax, **kwargs)

    fig.tight_layout()
    return fig


def plot_box(x, y, title='', ax=None, **kwargs):
    '''Plot box plot.

    Parameters
    ----------
    x : numpy.array of object
        values to group y. its unique values are labels for x-axis
    y : numpy.array of numeric
        values for y-axis
    title : str
        title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.
    kwargs : dict
        keyword arguments passing to :func:`matplotlib.pyplot.boxplot`

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()

    uniq = np.unique(x)
    values = [y[x == i] for i in uniq]
    ax.boxplot(values, labels=uniq, **kwargs)
    # plot significance bars
    n = len(uniq)
    annot = []
    for i, j in combinations(range(n), 2):
        y1 = values[i]
        y2 = values[j]
        test = stats.ttest_ind(y1, y2, equal_var=False)
        annot.append('{0} vs. {1}: {2:.3f}'.format(
            uniq[i], uniq[j], test.pvalue))
    ax.annotate('\n'.join(annot), xycoords=ax.transAxes,
                xy=(.5, 1), ha='center', va='top',
                fontsize='x-small',
                # fc, ec: face/edge color
                bbox=dict(boxstyle="round", fc="0.8", ec="none", alpha=0.7))
    ax.set_title(title)
    return ax


def plot_scatter(x, y, title='', ax=None, **kwargs):
    '''Plot scatter plot.

    Parameters
    ----------
    x : numeric
        values for x-axis
    y : numeric
        values for y-axis
    title : str
        title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.
    kwargs : dict
        keyword arguments passing to :func:`matplotlib.pyplot.scatter`

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.

    '''
    if ax is None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()

    r, p = stats.spearmanr(x, y)
    if r > 0:
        c = 'green'
    else:
        c = 'red'
    fit = np.polyfit(x, y, deg=1)
    ax.plot(x, fit[0] * x + fit[1], color=c)
    ax.scatter(x, y, alpha=0.2, color=c, **kwargs)
    ax.annotate("r={0:.2f} p={1:.3f}".format(r, p), xycoords=ax.transAxes,
                xy=(.5, 1), ha='center', va='top',
                fontsize='small',
                bbox=dict(boxstyle="round", fc="0.8", ec="none", alpha=0.7))
    ax.set_title(title)
    return ax
