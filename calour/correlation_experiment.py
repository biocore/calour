'''
correlation experiment (:mod:`calour.correlation_experiment`)
=======================================================

.. currentmodule:: calour.correlation_experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   CorrelationExperiment
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
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests

from .experiment import Experiment
from .util import _to_list
from .analysis import _new_experiment_from_pvals, _CALOUR_DIRECTION, _CALOUR_STAT

logger = getLogger(__name__)


class CorrelationExperiment(Experiment):
    '''This class stores a correlation matrix data and corresponding analysis methods.
    Besides the main data matrix (which is the correlation values) it also stores an additional Experiment (in self.qvals) that contains a matrix containing the q-values for each correlation.
    These can be plotted on top of the correlation matrix to show the significance of each correlation.

    This is a child class of :class:`.Experiment`.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The Correlation values (between -1 and 1)
    sample_metadata : pandas.DataFrame
        The metadata on the samples (rows in the matrix, shown in columns in the heatmap)
    feature_metadata : pandas.DataFrame
        The metadata on the features (columns in the matrix, shown in rows in the heatmap)
    qvals : numpy.ndarray or scipy.sparse.csr_matrix or None
        The q-values for the correlation values
        NOTE: This is not guaranteed to be in the same order as the data matrix (unless _sync_qvals() is called)
    description : str
        name of experiment
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`
    databases: iterable of str, optional
        database interface names to show by default in heatmap() function
        by default use None (no databases)
        For ASV correlations, can use 'dbbact'
        For gene correlations, can use 'mrna'

    Attributes
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The log ratio table for OTUs or ASVs.
        Samples are in row and features in column. values are float (can be negative)
        with np.nan indicating ratio for the specific feature does not exist.
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    qvals: numpy.ndarray or scipy.sparse.csr_matrix or None
        The q-values for the correlation values
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or dense numpy array.
    info : dict
        information about the experiment (data md5, filenames, etc.)
    description : str
        name of the experiment
    databases : dict
        keys are the database names (i.e. 'dbbact' / 'gnps')
        values are the database specific data for the experiment (i.e. annotations for dbbact)

    See Also
    --------
    Experiment
    '''
    def __init__(self, *args, qvals=None, **kwargs):
        '''Init the CorrelationExperiment class
        By default we set sparse=False (as we usually have a dense matrix)
        '''
        if 'sparse' not in kwargs:
            kwargs['sparse'] = False
        super().__init__(*args, **kwargs)
        if qvals is not None:
            if self.data.shape != qvals.shape:
                raise ValueError('qvals shape %s does not match data shape %s' % (qvals.shape, self.data.shape))
            self.qvals = Experiment(data=qvals, sample_metadata=self.sample_metadata, feature_metadata=self.feature_metadata, sparse=self.sparse)

    def _sync_qvals(self):
        '''Sync the q-values experiment with the main experiment
        Used to make sure the q-values are in the same order as the data matrix.
        '''
        self.qvals = self.qvals.filter_ids(self.feature_metadata.index, axis='f')
        self.qvals = self.qvals.filter_ids(self.sample_metadata.index, axis='s')

    def _get_abundance_info(self, row:int , col:int):
        '''Get a string with the abundance information for display in the interactive heatmap
        Also returns the qvalue if it exists.

        Parameters
        ----------
        row : int
            The row index
        col : int
            The column index

        Returns
        -------
        str
            The string with the abundance information
        '''
        if self.qvals is None:
            qval = 'NA'
        else:
            qval = self.qvals.data[row, col]
        return '{:.2E}, qval: {:.2f}'.format(self.data[row, col], qval)
    
    def heatmap(self, significance_plot=['cmap'],significance_threshold=0.05, significance_plot_params={'color': 'red'}, cmap='bwr', *args, **kwargs):
        '''Plot a heatmap for the ratio experiment.
        The heatmap includes indication for significant correlations. This can be as a different set of colors for the significant correlations or by plotting a marker for the significant correlations.

        This method accepts the same parameters as input with its parent class method.
        In addition, it accepts the following parameters:
        significance_plot : list of str, optional
            The type of significance plot to show. Can be 'cmap' and/or 'x'
        significance_threshold : float, optional
            The threshold for the q-values to be considered significant.
        significance_plot_params : dict, optional
            The parameters to be passed to the plot function for the significance values.
            If 'cmap' is in the list, use the 'cmap' parameter in significance_plot_params to set the colormap for the significant values.
            If 'x' is in the list, use the 'significance_plot_params' parameter to set the plot parameters for the significance values.

        See Also
        --------
        Experiment.heatmap

        '''
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        if 'clim' not in kwargs:
            min_val = np.min(self.get_data()[:])
            max_val = np.max(self.get_data()[:])
            range_val = np.max([np.abs(min_val), np.abs(max_val)])
            kwargs['clim'] = (-range_val, range_val)

        if significance_plot is None or significance_plot == []:
            if self.qvals is None:
                raise ValueError('No qvals attached to experiment. Please provide a qvals matrix to plot the significance values or use significance_plot=[] to not plot significance values.')
        else:
            self._sync_qvals()

        data_changed = False
        if 'cmap' in significance_plot:
            # copy the data
            old_data = self.get_data(copy=True)
            data_changed = True

            # eps is added to the data to avoid overlap in the colormaps for significant/non-significant values
            eps = 1e-7
            
            max_val = kwargs['clim'][1]
            min_val = kwargs['clim'][0]
            self.data[self.data>max_val]=max_val
            self.data[self.data<min_val]=min_val
            self.data = self.data - (max_val + eps)

            qv = self.qvals.get_data(sparse=False)
            sig_pos = qv < significance_threshold
            self.data[sig_pos]+= (2*max_val)+eps
            if 'cmap' in significance_plot_params:
                cmap_sig = significance_plot_params['cmap']
                del significance_plot_params['cmap']
            else:
                cmap_sig = 'PiYG'

            # create the colormap which is a concatenation of the original colormap and the significant colormap
            colors_nonsig = plt.get_cmap(cmap)(np.linspace(0, 1, 128))
            colors_sig = plt.get_cmap(cmap_sig)(np.linspace(0, 1, 128))
            colors = np.vstack((colors_nonsig, colors_sig))
            concatenated_cmap = LinearSegmentedColormap.from_list('concatenated_cmap', colors)
            kwargs['cmap'] = concatenated_cmap
            # adjust the clim to account for the added values (negative values are for the non-significant values, positive values are for the significant values)
            kwargs['clim'] = (2*kwargs['clim'][0], 2*kwargs['clim'][1])

        # call the heatmap function from the parent class using the exp object
        ax = super().heatmap(*args, **kwargs)

        # if the data was changed (for the significance plot), revert it back to the original data
        if data_changed:
            self.data = old_data

        # add the significant correlations plot
        if 'x' in significance_plot:
            if self.qvals is not None:
                qv = self.qvals.get_data(sparse=False)
                show_pos = np.where(qv < significance_threshold)
                for i, j in zip(*show_pos):
                    ax.plot([i-0.5, i+0.5], [j-0.5, j+0.5], **significance_plot_params)
                    ax.plot([i-0.5, i+0.5], [j+0.5, j-0.5], **significance_plot_params)

        return ax
    
    def save(self, prefix, **kwargs):
        '''Save the correlation experiment to a file
        overwrites the save function in Experiment to also save the q-values (as a new experiment named prefix+"_qvals").

        Parameters
        ----------
        prefix : str
            file path (suffixes auto added for the 3 files) to save to.
        **kwargs : dict
            Additional arguments to pass to the Experiment.save() function
        '''
        self._sync_qvals()
        super().save(prefix, **kwargs)
        if self.qvals is not None:
            self.qvals.save(prefix+'_qvals', **kwargs)
            logger.debug('Saved qvals experiment to %s_qvals' % prefix)
        else:
            logger.warning('No qvals attached to experiment. qvals experiment not saved')

    def _calculate_corr_matrix(df1, df2):
        '''Calculate the spearman correlation matrix between all columns of two DataFrames
        Ignores non-numeric values

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to calculate the correlation matrix for
            
        Returns
        -------
        corrs : numpy.ndarray
            The correlation matrix
            pvals : numpy.ndarray
            The p-values for the correlation matrix
        '''
        pvals=np.ones([len(df1.columns),len(df2.columns)])
        corrs=np.zeros([len(df1.columns),len(df2.columns)])
        for idx1,r in enumerate(df1.columns):
            for idx2,c in enumerate(df2.columns):
                c1=df1[r].values
                c2=df2[c].values
                try:
                    ccor = scipy.stats.spearmanr(c1,c2,nan_policy='omit')
                    pvals[idx1][idx2] = ccor.pvalue
                    corrs[idx1][idx2] = ccor.correlation
                    if np.isnan(ccor.correlation):
                        pvals[idx1][idx2] = 1
                        corrs[idx1][idx2] = 0
                except:
                    pvals[idx1][idx2] = 1
                    corrs[idx1][idx2] = 0
        return corrs,pvals

    @classmethod
    def from_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame|None = None):
        '''Create a CorrelationExperiment from a pandas DataFrame (such as the experiment sample_metadata)
        Calculates the correlations between all dataframe columns

        Parameters
        ----------
        df1 : pandas.DataFrame
            The first DataFrame to calculate the correlation matrix for
        df2 : pandas.DataFrame
            The second DataFrame to calculate the correlation matrix for
            If None, will use df1

        Returns
        -------
        CorrelationExperiment
            The correlation experiment
        '''
        if df2 is None:
            df2=df1
        corrs,pvals = self._calculate_corr_matrix(df1, df2)
        new_smd = pd.DataFrame(index=df1.columns)
        new_fmd = pd.DataFrame(index=df2.columns)
        new_smd['SampleID']=new_smd.index.values
        new_fmd['_feature_id']=new_fmd.index.values
        exp=CorrelationExperiment(data=corrs, sample_metadata=new_smd, feature_metadata=new_fmd, qvals=pvals, sparse=False)
        exp=exp.cluster_data(axis='f')
        exp=exp.cluster_data(axis='s')
        return exp
