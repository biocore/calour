'''
experiment (:mod:`calour.experiment`)
=====================================

.. currentmodule:: calour.experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   Experiment

'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy, copy
from functools import wraps
from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.sparse

from .util import _convert_axis_name

logger = getLogger(__name__)


class Experiment:
    '''This class contains the data for a experiment or a meta-experiment.

    The data set includes 3 aligned tables: a data table (otu table,
    gene table, metabolomic table, or all those tables combined), a
    sample metadata table, and a feature metadata table.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata for the samples
    feature_metadata : pandas.DataFrame
        The metadata for the features
    description : str
        name of experiment
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`
    databases: iterable of str, optional
        database interface names to show by default in heatmap() function

    Attributes
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`
    normalized : int
        the normalization factor. it is zero if not normalized
    info : dict
        information about the experiment (data md5, filenames, etc.)
    description : str
        a short description of the experiment
    databases : defaultdict(dict)
        keys are the database names (i.e. 'dbbact' / 'gnps')
        values are the database specific data for the experiment (i.e. annotations for dbbact)

    See Also
    --------
    AmpliconExperiment
    MS1Experiment
    '''
    def __init__(self, data, sample_metadata, feature_metadata=None, databases=(),
                 info=None, description='', sparse=True):
        self.data = data
        self.sample_metadata = sample_metadata
        if feature_metadata is None:
            feature_metadata = pd.DataFrame(np.arange(data.shape[1]))
        self.feature_metadata = feature_metadata
        self.validate()
        self.info = {} if info is None else info
        self.description = description
        self.normalized = 0
        # the function calling history list
        self._call_history = []
        # whether to log to calling history
        self._log = True

        # flag if data array is sparse (True) or dense (False)
        self.sparse = sparse

        # the database local specific data (to use for feature information)
        self.databases = defaultdict(dict)
        for cdatabase in databases:
            self.databases[cdatabase] = {}

    def validate(self):
        '''Validate the Experiment object.

        This simply checks the shape of data table with
        sample_metadata and feature_metadata.

        Raises
        ------
        ValueError
            If the shapes of the 3 tables do not agree.
        '''
        duplicates = self.sample_metadata.index.duplicated(keep=False)
        if duplicates.any():
            raise ValueError(
                'Duplicate sample IDs exist in positions %s.' % np.where(duplicates)[0])
        duplicates = self.feature_metadata.index.duplicated(keep=False)
        if duplicates.any():
            raise ValueError(
                'Duplicate feature IDs exist in positions %s.' % np.where(duplicates)[0])

        n_sample, n_feature = self.data.shape
        ns = self.sample_metadata.shape[0]
        nf = self.feature_metadata.shape[0]
        if n_sample != ns:
            raise ValueError(
                'data table must have the same number of samples with sample_metadata table (%d != %d).' % (n_sample, ns))
        if n_feature != nf:
            raise ValueError(
                'data table must have the same number of features with feature_metadata table (%d != %d).' % (n_feature, nf))
        return ns, nf

    @property
    def shape(self):
        return self.validate()

    @property
    def sparse(self):
        return scipy.sparse.issparse(self.data)

    @sparse.setter
    def sparse(self, sparse):
        if sparse is True and not scipy.sparse.issparse(self.data):
            self.data = scipy.sparse.csr_matrix(self.data)
        elif sparse is False and scipy.sparse.issparse(self.data):
            self.data = self.data.toarray()

    def __repr__(self):
        '''Return a string representation of this object.
        The form is: class (description) with X samples, Y features
        '''
        l1 = self.__class__.__name__
        if self.description:
            l1 += ' ("%s")' % self.description
        l1 += ' with %d samples, %d features' % self.data.shape
        return l1

    def __eq__(self, other):
        '''Check equality.

        It compares ``data``, ``sample_metadata``, and
        ``feature_metadata`` attributes.  to check sparsity and do
        the conversion if needed first.
        '''
        if self.sparse is True:
            data = self.data.toarray()
        else:
            data = self.data
        if other.sparse is True:
            other_data = other.data.toarray()
        else:
            other_data = other.data
        return (np.array_equal(data, other_data)
                and pd.DataFrame.equals(self.feature_metadata, other.feature_metadata)
                and pd.DataFrame.equals(self.sample_metadata, other.sample_metadata))

    def __ne__(self, other):
        return not (self == other)

    def __getitem__(self, pos):
        '''Get the value from data table for (sample_id, feature_id)

        Parameters
        ----------
        pos : tuple of (str, str)
            the SampleID, FeatureID

        Returns
        -------
        float
            The value of feature ID in sample ID
        '''
        if not isinstance(pos, tuple) or len(pos) != 2:
            raise SyntaxError('Must supply sample ID, feature ID')

        sample = pos[0]
        feature = pos[1]
        if isinstance(sample, slice):
            sample_pos = sample
        else:
            try:
                sample_pos = self.sample_metadata.index.get_loc(sample)
            except KeyError:
                raise KeyError('SampleID %s not in experiment samples' % sample)
        if isinstance(feature, slice):
            feature_pos = feature
        else:
            try:
                feature_pos = self.feature_metadata.index.get_loc(feature)
            except KeyError:
                raise KeyError('FeatureID %s not in experiment features' % feature)
        if self.sparse:
            dat = self.get_data(sparse=False)
        else:
            dat = self.get_data()
        return dat[sample_pos, feature_pos]

    def copy(self):
        '''Copy the object (deeply).

        It calls :func:`Experiment.__deepcopy__` to make copy.

        Returns
        -------
        Experiment

        '''
        return deepcopy(self)

    def __deepcopy__(self, memo):
        '''Implement the deepcopy since pandas has problem deepcopy empty dataframe

        When using the default deepcopy on an empty dataframe (columns but no rows), we get an error.
        This happens when dataframe has 0 rows in pandas 0.19.2 np112py35_1.
        So we manually use copy instead of deepcopy for empty dataframes
        '''
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, deepcopy(v, memo))
            except:
                logger.debug('Failed to copy attribute %r, doing shallow copy on it' % k)
                setattr(result, k, copy(v))
                memo[id(k)] = v
        return result

    @staticmethod
    def _record_sig(func):
        '''Record the function calls to history.

        Note this requires the function decorated to return an
        :class:`.Experiment` object.
        '''
        fn = func.__qualname__

        @wraps(func)
        def inner(*args, **kwargs):
            # this extra code here is to prevent recording func call
            # if the method is called inside another method.
            exp = args[0]
            log = exp._log
            try:
                logger.debug('Run func {}'.format(fn))
                exp._log = False
                new_exp = func(*args, **kwargs)
                new_exp._log = log
                if log is True:
                    # do not use `'%r' % i` because it causes error when i is a tuple
                    param = ['{!r}'.format(i) for i in args[1:]] + ['{0!s}={1!r}'.format(k, v) for k, v in kwargs.items()]
                    param = ', '.join(param)
                    new_exp._call_history.append('{0}({1})'.format(fn, param))
                    logger.debug('Current object: {}'.format(new_exp))
            finally:
                # set log status back
                exp._log = log
            return new_exp

        return inner

    def get_data(self, sparse=None, copy=False):
        '''Get the data as a 2d array

        Get the data 2d array (each column is a feature and row is a sample)

        Parameters
        ----------
        sparse : bool, default=None
            default to pass original data (sparse or dense).
            True to get as sparse. False to get as dense.
        copy : bool, default=False
            True to get a copy of the data; otherwise, it can be
            the original data or a copy (default).

        Returns
        -------
        ``Experiment.data``
        '''
        if sparse is None:
            if copy:
                return self.data.copy()
            else:
                return self.data
        elif sparse:
            if self.sparse:
                if copy:
                    return self.data.copy()
                else:
                    return self.data
            else:
                return scipy.sparse.csr_matrix(self.data)
        else:
            if self.sparse:
                return self.data.toarray()
            else:
                if copy:
                    return self.data.copy()
                else:
                    return self.data

    def to_pandas(self, sample_field=None, feature_field=None, sparse=None):
        '''Convert Experiment object to a pandas DataFrame.

        Samples are rows and features are columns. You can specify the metadata fields
        for the index (default is sample_metadata index) and column labels
        (default is feature_metadata index).

        Parameters
        ----------
        sample_field : str or None, optional
            Column name of the sample_metadata to use as the index for the resulting pandas DataFrame.
            None (default) is the sample_metadata index
        feature_field : str or None, optional
            Column name of the feature_metadata to use for column labels for the resulting pandas DataFrame.
            None (default) is the feature_metadata index
        sparse: bool or None, optional
            None (default) to get sparsity based on the underlying Experiment sparsity.
            True to force to sparse pandas.DataFrame;
            False to force to standard pandas.DataFrame

        Returns
        -------
        pandas.Dataframe
        '''
        if sample_field is None:
            ind = self.sample_metadata.index
        else:
            ind = self.sample_metadata[sample_field]
        if feature_field is None:
            cols = self.feature_metadata.index
        else:
            cols = self.feature_metadata[feature_field]

        if self.sparse and sparse:
            df = pd.DataFrame.sparse.from_spmatrix(self.data, index=ind, columns=cols)
        elif self.sparse:
            df = pd.DataFrame(self.data.todense(), index=ind, columns=cols)
        elif sparse:
            df = pd.DataFrame(scipy.sparse.csr_matrix(self.data), index=ind, columns=cols)
        else:
            df = pd.DataFrame(self.data, index=ind, columns=cols, copy=True)
        return df

    @classmethod
    def from_pandas(cls, df, exp=None):
        '''Convert a Pandas DataFrame into an experiment.

        It take an existing Calour Experiment object (if supplied) to
        obtain its feature and sample metadata while replacing the
        data with the values from the pandas dataframe. Note currently
        only works with non-sparse DataFrame

        Parameters
        ----------
        df : Pandas.DataFrame
            The dataframe to use. should contain samples in rows and features in columns.
            Index values will be used for the sample_metadata index and column names will be used for feature_metadata index
        exp : Experiment, optional
            If not None, use sample and feature metadata from the experiment

        Returns
        -------
        Experiment
            with non-sparse data

        '''
        if exp is None:
            sample_metadata = pd.DataFrame(index=df.index)
            sample_metadata['id'] = sample_metadata.index
            feature_metadata = pd.DataFrame(index=df.columns)
            feature_metadata['id'] = feature_metadata.index
            info = {}
            description = 'From Pandas DataFrame'
        else:
            description = exp.description + ' From Pandas'
            info = exp.info
            sample_metadata = exp.sample_metadata.loc[df.index.values, ]
            feature_metadata = exp.feature_metadata.loc[df.columns.values, ]
            cls = exp.__class__

        newexp = cls(df.values, sample_metadata, feature_metadata,
                     info=info, description=description, sparse=False)
        return newexp

    @_convert_axis_name
    def iterate(self, field=None, axis=0):
        '''Iterate over all sample or feature groups.

        Iterate over all unique values of metadata field. Each
        iteration yields an experiment with all samples containing the
        value. If field is None, iterate over all samples or features
        one at a time.

        Parameters
        ----------
        field: str or None, optional
            If None, each iteration yields an Experiment with a single
            sample or feature.  If not None, iterate over all unique
            values of metadata field, with each iteration yielding an
            Experiment with all samples/features with this value.
        axis: int or str, optional
            The axis on which to iterate.
            0 or 's' to iterate over samples
            1 or 'f' to iterate over features

        Yields
        -------
        str
            current value of the field (or the _sample_id/_feature_id (for axis='s'/'f' respectively) if field==None)
        Experiment
            With all samples or features containing the current value in field (or a single sample if field=None)

        '''
        if axis == 0:
            metadata = self.sample_metadata
            if field is None:
                field = '_sample_id'
        else:
            metadata = self.feature_metadata
            if field is None:
                field = '_feature_id'

        vals = metadata[field].unique()
        for cval in vals:
            yield cval, self.filter_by_metadata(field, [cval], axis=axis)
