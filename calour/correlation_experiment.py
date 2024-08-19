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

    def heatmap(self, show_significance=True, significance_threshold=0.05, significance_plot_params={'color': 'red'},*args, **kwargs):
        '''Plot a heatmap for the ratio experiment.

        This method accepts the same parameters as input with
        its parent class method.
        In addition, it accepts the following parameters:
        show_significance : bool, optional
            If True, the q-values will be plotted on top of the heatmap.
        significance_threshold : float, optional
            The threshold for the q-values to be considered significant.
        significance_plot_params : dict, optional
            The parameters to be passed to the plot function for the significance values.

        See Also
        --------
        Experiment.heatmap

        '''
        if 'clim' not in kwargs:
            min_val = np.min(self.get_data()[:])
            max_val = np.max(self.get_data()[:])
            range_val = np.max([np.abs(min_val), np.abs(max_val)])
            kwargs['clim'] = (-range_val, range_val)
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'coolwarm'

        ax = super().heatmap(*args, **kwargs)
        if show_significance:
            if self.qvals is not None:
                self._sync_qvals()
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
        super().save(prefix, **kwargs)
        if self.qvals is not None:
            self.qvals.save_biom(prefix+'_qvals.biom')
            logger.debug('Saved qvals experiment to %s_qvals.biom' % prefix)
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


    # def save(self, filename, **kwargs):
    #     '''Save the correlation experiment to a file

    #     Parameters
    #     ----------
    #     filename : str
    #         The file to save the experiment to
    #     **kwargs : dict
    #         Additional arguments to pass to the save
    #     '''
    #     super().save(filename, **kwargs)
    #     if self.qvals is not None:
    #         self.qvals(filename+'.qvals', **kwargs)


    @classmethod
    def read_correlation(self, filename, **kwargs):
        '''Read the correlation experiment from a file

        Parameters
        ----------
        filename : str
            The file to read the experiment from
        **kwargs : dict
            Additional arguments to pass to the read
        '''
        from .io import read

        if 'normalize' not in kwargs:
            kwargs['normalize'] = None

        exp = read(filename+'.biom', sample_metadata_file=filename+'_sample.txt', feature_metadata_file=filename+'_feature.txt', cls=CorrelationExperiment, **kwargs)

        exp.qvals = read(filename+'_qvals.biom', sample_metadata_file=filename+'_qvals_sample.txt', feature_metadata_file=filename+'_qvals_feature.txt', **kwargs)
        return exp

    # @classmethod
    # def from_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame = None):
    #     '''Create a CorrelationExperiment from a pandas DataFrame (such as the experiment sample_metadata)
    #     Calculates the correlations between all dataframe columns

    #     Parameters
    #     ----------
    #     df1 : pandas.DataFrame
    #         The first DataFrame to calculate the correlation matrix for
    #     df2 : pandas.DataFrame
    #         The second DataFrame to calculate the correlation matrix for
    #         If None, will use df1

    #     Returns
    #     -------
    #     CorrelationExperiment
    #         The correlation experiment
    #     '''
    #     if df2 is None:
    #         df2=df1
    #     corrs,pvals = self._calculate_corr_matrix(df1, df2)
    #     new_smd = pd.DataFrame(index=df1.columns)
    #     new_fmd = pd.DataFrame(index=df2.columns)
    #     new_smd['SampleID']=new_smd.index.values
    #     new_fmd['_feature_id']=new_fmd.index.values
    #     exp=CorrelationExperiment(data=corrs, sample_metadata=new_smd, feature_metadata=new_fmd, qvals=pvals, sparse=False)
    #     exp=exp.cluster_data(axis='f')
    #     exp=exp.cluster_data(axis='s')
    #     return exp

    # @classmethod
    # def from_data(self, corr, samples, features, qvals):
    #     '''Create a CorrelationExperiment from a numpy array and metadata

    #     Parameters
    #     ----------
    #     corr : numpy.ndarray
    #         The correlation matrix
    #     samples : list or pandas.DataFrame
    #         The sample metadata
    #     features : list or pandas.DataFrame
    #         The feature metadata
    #     qvals : numpy.ndarray
    #         The q-value matrix for the correlations

    #     Returns
    #     -------
    #     CorrelationExperiment
    #         The correlation experiment
    #     '''
    #     if isinstance(samples, list):
    #         samples=pd.DataFrame(index=samples)
    #     if isinstance(features, list):
    #         features=pd.DataFrame(index=features)
    #     if 'SampleID' not in samples.columns:
    #         samples['SampleID']=samples.index.values
    #     if '_feature_id' not in features.columns:
    #         features['_feature_id']=features.index.values

    #     return CorrelationExperiment(data=corr, sample_metadata=samples, feature_metadata=features, qvals=qvals, sparse=False)
