'''
export_html (:mod:`calour.export_html`)
=======================================

.. currentmodule:: calour.export_html

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   export_html
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from io import BytesIO
import urllib
from pkg_resources import resource_filename

import numpy as np
import matplotlib as mpl

from . import Experiment
from .heatmap.heatmap import _transition_index

logger = getLogger(__name__)


def _list_to_string(cl):
    '''Convert a list to a string representation of the list

    For use in the html javascript

    Parameters
    ----------
    cl : list

    Returns
    -------
    str
        ready for embedding into html javascript
    '''
    cstr = '['
    for cval in cl:
        cstr += '"'
        cstr += str(cval)
        cstr += '",'
    cstr = cstr[:-1] + ']'
    return cstr


def export_html(exp: Experiment, sample_field=None, feature_field=None, title=None,
                xticklabel_len=50, cmap=None, clim=(None, None), norm=mpl.colors.LogNorm(),
                output_file='out', html_template=None, bad_color='black'):
    '''Export an interactive html heatmap for the experiment.

    Creates a standalone html file with interactive d3.js heatmap of the experiment and interface to dbBact.

    Parameters
    ----------
    sample_field : str or None, optional
        The field to display on the x-axis (sample).
        None (default) to not show x labels.
        str to display field values for this field.
    feature_field : str or None, optional
        Name of the field to display on the y-axis (features) or None not to display names
    title : None or str (optional)
        None (default) to show experiment description field as title. str to set title to str.
    xticklabel_len : int, optional or None
        The maximal length for the x label strings (will be cut to
        this length if longer). Used to prevent long labels from
        taking too much space. None indicates no cutting
    cmap : None or str, optional
        None (default) to use mpl default color map. str to use colormap named str.
    clim : tuple of (float, float) or None, optional
        the min and max values for the heatmap or None to use all range. It uses the min
        and max values in the ``data`` array by default.
    norm : matplotlib.colors.Normalize or None
        passed to ``norm`` parameter of matplotlib.pyplot.imshow. Default is log scale.
    output_file : str, optional
        Name of the output html file (no .html ending - it will be appended).
    html_template : str or None, optional
        Name of the html template to use. None to use the default export_html_template.html template
    bad_color: str or matplotlib color, optional
        The heatmap color to use for masked / NaN values
    '''
    import matplotlib.pyplot as plt

    if html_template is None:
        html_template = resource_filename(
            __package__, 'export_html_template.html')
        logger.debug('using default template file %s' % html_template)

    logger.debug('export_html heatmap')

    numrows, numcols = exp.shape
    data = exp.get_data(sparse=False)

    # step 2. plot heatmap.
    # init the default colormap
    if cmap is None:
        cmap = plt.rcParams['image.cmap']
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # this set cells of zero value.
    cmap.set_bad(bad_color)

    # plot the heatmap with 1 pixel per feature/sample, no axes/lines
    fig = plt.figure(frameon=False, dpi=300)
    fig.set_size_inches(exp.shape[0] / 300, exp.shape[1] / 300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # plot the heatmap
    vmin, vmax = clim
    # logNorm requires positive values. set it to default None if vmin is zero
    if vmin == 0:
        vmin = None
    ax.imshow(data.transpose(), interpolation='nearest', vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)

    if title is None:
        title = exp.description

    # add parameters to html template
    with open(html_template) as fl:
        html_page = fl.read()
    html_page = html_page.replace('// ids go here', 'var ids = %s;' %
                                  _list_to_string(exp.feature_metadata.index.values))
    html_page = html_page.replace('// samples go here', 'var samples = %s;' %
                                  _list_to_string(exp.sample_metadata.index.values))
    if feature_field is not None:
        html_page = html_page.replace('// yticklabels go here', 'var yticklabels = %s;' %
                                      _list_to_string(exp.feature_metadata[feature_field].values))
    if sample_field is not None:
        html_page = html_page.replace('// field_name goes here', 'var field_name = "%s";' %
                                      sample_field)
    html_page = html_page.replace('// title_text goes here', 'var title_text = "%s";' %
                                  title)

    # add vertical lines between sample groups and add x tick labels
    if sample_field is not None:
        try:
            xticks = _transition_index(exp.sample_metadata[sample_field])
        except KeyError:
            raise ValueError('Sample field %r not in sample metadata.' %
                             sample_field)
        x_pos, x_val = zip(*xticks)
        x_pos = np.array([0.] + list(x_pos))

        html_page = html_page.replace('// vlines go here', 'var vlines = %s;' %
                                      _list_to_string(x_pos[1:-1]))
        xtick_pos = x_pos[:-1] + (x_pos[1:] - x_pos[:-1]) / 2
        html_page = html_page.replace('// xtick_pos go here', 'var xtick_pos = %s;' %
                                      _list_to_string(xtick_pos))

        xticklabels = [str(i) for i in x_val]
        # shorten x tick labels that are too long:
        if xticklabel_len is not None:
            mid = int(xticklabel_len / 2)
            xticklabels = ['%s..%s' % (i[:mid], i[-mid:])
                           if len(i) > xticklabel_len else i
                           for i in xticklabels]
        html_page = html_page.replace('// xtick_labels go here', 'var xtick_labels = %s;' %
                                      _list_to_string(xticklabels))

    # embed the figure png into the html page
    with BytesIO() as figfile:
        fig.savefig(figfile, format='png', dpi=300)
        figfile.seek(0)  # rewind to beginning of file
        import base64
        figdata_png = base64.b64encode(figfile.getvalue())
        figdata_png = urllib.parse.quote(figdata_png)
    html_page = html_page.replace('**image_goes_here**', figdata_png)

    if output_file[-5:] != '.html':
        output_file = output_file + '.html'

    # save the output html export
    with open(output_file, 'w') as fl:
        fl.write(html_page)
    logger.info('exported experiment to html file %s' % output_file)
    return
