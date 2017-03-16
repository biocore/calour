'''
ms1 experiment (:mod:`calour.ms1_experiment`)
=======================================================

.. currentmodule:: calour.ms1_experiment

Classes
^^^^^^^^
.. autosummary::
   :toctree: generated

   Ms1Experiment
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger

import numpy as np

from .experiment import Experiment
from .util import _get_taxonomy_string, _to_list


logger = getLogger(__name__)


class Ms1Experiment(Experiment):
    '''This class contains the data for a Mass-Spec ms1 spectra experiment or a meta experiment.

    Parameters
    ----------
    data : :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : :class:`pandas.DataFrame`
        The metadata on the samples
    feature_metadata : :class:`pandas.DataFrame`
        The metadata on the features
    description : str
        name of experiment
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`

    Attributes
    ----------
    data : :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : :class:`pandas.DataFrame`
        The metadata on the samples
    feature_metadata : :class:`pandas.DataFrame`
        The metadata on the features
    exp_metadata : dict
        metadata about the experiment (data md5, filenames, etc.)
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or numpy array.
    description : str
        name of the experiment

    See Also
    --------
    Experiment
    '''
    def __repr__(self):
        '''Return a string representation of this object.'''
        return 'Ms1Experiment %s with %d samples, %d features' % (
            self.description, self.data.shape[0], self.data.shape[1])

    def prepare_gnps(self):
        if '_calour_metabolomics_gnps_table' not in self.exp_metadata:
            logger.warn('No GNPS data file supplied - labels will be NA')
            self.feature_metadata['gnps'] = 'NA'
            return
        logger.debug('Adding gnps terms as "gnps" column in feature metadta')
        self.add_terms_to_features('gnps', use_term_list=None, field_name='gnps')
        logger.debug('Added terms')

    def plot(self, databases=('gnps',), feature_field='gnps', **kwargs):
        if feature_field == 'gnps':
            if 'gnps' not in self.feature_metadata.columns:
                logger.warn('no gnps feature metadata present. Plase run ".prepare_gnps()" to see gnps labels.')
                feature_field = 'id'
        # plot the experiment using taxonmy field and dbbact database
        super().plot(feature_field=feature_field, databases=databases, **kwargs)

    @Experiment._record_sig
    def sort_mz(exp, inplace=False):
        '''Sort the features based on the taxonomy

        Sort features based on the taxonomy (alphabetical)

        Parameters
        ----------
        inplace : bool (optional)
            False (default) to create a copy
            True to Replace data in exp
        Returns
        -------
        exp : Experiment
            sorted by taxonomy
        '''
        logger.debug('sorting by taxonomies')
        taxonomy = _get_taxonomy_string(exp, remove_underscore=True)
        sort_pos = np.argsort(taxonomy, kind='mergesort')
        exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
        return exp

    def plot_sort(self, fields=None, feature_field='gnps', sample_color_bars=None, feature_color_bars=None,
                  gui='cli', databases=('gnps',), color_bar_label=True, **kwargs):
        '''Plot bacteria after sorting by field

        This is a convenience wrapper for plot()

        Parameters
        ----------
        fields : str or list of str or None (optional)
            The field to sort samples by before plotting
            If list of str, sort by each field according to order in list
            if None, do not sort
        sample_color_bars : list, optional
            list of column names in the sample metadata. It plots a color bar
            for each column. It doesn't plot color bars by default (``None``)
        feature_color_bars : list, optional
            list of column names in the feature metadata. It plots a color bar
            for each column. It doesn't plot color bars by default (``None``)
        color_bar_label : bool, optional
            whether to show the label for the color bars
        gui : str or None, optional
            GUI to use:
            'cli' : simple command line gui
            'jupyter' : jupyter notebook interactive gui
            'qt5' : qt5 based interactive gui
            None : no interactivity - just a matplotlib figure
        databases : Iterable of str
            a list of databases to access or add annotation
        kwargs : dict, optional
            keyword arguments passing to :ref:`plot<plot-ref>` function.

        '''
        if feature_field == 'gnps':
            if 'gnps' not in self.feature_metadata.columns:
                logger.warn('no gnps feature metadata present. Plase run ".prepare_gnps()" to see gnps labels.')
                feature_field = 'id'
        if fields is not None:
            newexp = self.copy()
            fields = _to_list(fields)
            for cfield in fields:
                newexp.sort_samples(cfield, inplace=True)
            plot_field = cfield
        else:
            newexp = self
            plot_field = None
        if 'sample_field' in kwargs:
            newexp.plot(feature_field='ID', sample_color_bars=sample_color_bars, feature_color_bars=feature_color_bars,
                        gui=gui, databases=databases, color_bar_label=color_bar_label, **kwargs)
        else:
            newexp.plot(sample_field=plot_field, feature_field='ID', sample_color_bars=sample_color_bars, feature_color_bars=feature_color_bars,
                        gui=gui, databases=databases, color_bar_label=color_bar_label, **kwargs)
