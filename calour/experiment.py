# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy

from importlib import import_module
import inspect
from functools import wraps

import pandas as pd
import numpy as np
import scipy.sparse


logger = getLogger(__name__)


class Experiment:
    '''This class contains the data for a experiment or a meta experiment.

    The data set includes a data table (otu table, gene table,
    metabolomic table, or all those tables combined), a sample
    metadata table, and a feature metadata.

    Parameters
    ----------
    data : ``numpy.array`` or ``scipy.sparse``
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : ``pandas.DataFrame``
        The metadata on the samples
    feature_metadata : ``pandas.DataFrame``
        The metadata on the features
    description : str
        Text describing the experiment
    sparse : bool
        store the data array in sparse or dense matrix

    Attributes
    ----------
    data : ``numpy.array`` or ``scipy.sparse``
        The abundance table for OTUs, metabolites, genes, etc. Samples
        are in row and features in column
    sample_metadata : ``pandas.DataFrame``
        The metadata on the samples
    feature_metadata : ``pandas.DataFrame``
        The metadata on the features
    exp_metadata : dict
        metadata about the experiment (data md5, filenames, etc.)
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse or dense array.
    description : str
        description of the experiment
    '''
    def __init__(self, data, sample_metadata, feature_metadata=None,
                 exp_metadata={}, description='', sparse=True):
        self.data = data
        self.sample_metadata = sample_metadata
        self.feature_metadata = feature_metadata
        self.exp_metadata = exp_metadata
        self.description = description

        # the function calling history list
        self._call_history = []
        # whether to log to history
        self._log = True

        # flag if data array is sparse (True) or dense (False)
        self.sparse = sparse

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, sparse):
        if sparse is True and not scipy.sparse.issparse(self.data):
            self.data = scipy.sparse.csr_matrix(self.data)
        elif sparse is False and scipy.sparse.issparse(self.data):
            self.data = self.data.toarray()
        self._sparse = sparse

    def __repr__(self):
        '''Return a string representation of this object.'''
        return 'Experiment %s with %d samples, %d features' % (
            self.description, self.data.shape[0], self.data.shape[1])

    def __eq__(self, other):
        '''Check equality.

        Need to check sparsity and do the conversion if needed first.
        '''
        if self.sparse is True:
            data = self.data.toarray()
        else:
            data = self.data
        if other.sparse is True:
            other_data = other.data.toarray()
        else:
            other_data = other.data
        return (np.array_equal(data, other_data) and
                pd.DataFrame.equals(self.feature_metadata, other.feature_metadata) and
                pd.DataFrame.equals(self.sample_metadata, other.sample_metadata))

    def __ne__(self, other):
        return not (self == other)

    def __copy__(self):
        '''Return a copy of Experiment'''
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        '''Return a deep copy of Experiment. '''
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @staticmethod
    def _record_sig(func):
        '''Record the function calls to history. '''
        fn = func.__qualname__

        @wraps(func)
        def inner(*args, **kwargs):
            # this extra code here is to prevent recording func call
            # if the method is called inside another method.
            exp = args[0]
            log = exp._log
            try:
                new_exp = func(*args, **kwargs)
                if exp._log is True:
                    param = ['%r' % i for i in args[1:]] + ['%s=%r' % (k, v) for k, v in kwargs.items()]
                    param = ', '.join(param)
                    new_exp._call_history.append('{0}({1})'.format(fn, param))
                    exp._log = False
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
        sparse : None or bool (optional)
            None (default) to pass original data format (sparse or dense).
            True to get as sparse.
            False to get as dense
        copy : bool (optional)
            True (default) to get a copy of the data
            False to get the original data (for inplace)
        '''
        if sparse is None:
            if copy:
                return self.data.copy()
            else:
                return self.data
        elif sparse:
            if scipy.sparse.issparse(self.data):
                if copy:
                    return self.data.copy()
                else:
                    return self.data
            else:
                return scipy.sparse.csr_matrix(self.data)
        else:
            if scipy.sparse.issparse(self.data):
                return self.data.toarray()
            else:
                if copy:
                    return self.data.copy()
                else:
                    return self.data

    @property
    def shape(self):
        '''Get the number of samples by features in the experiment. '''
        return self.get_data().shape

    def reorder(self, new_order, axis=0, inplace=False):
        '''Reorder according to indices in the new order.

        Note that we can also drop samples in new order.

        Parameters
        ----------
        new_order : Iterable of int or boolean mask
            the order of new indices
        axis : 0 for samples or 1 for features
            the axis where the reorder occurs
        inplace : bool, optional
            reorder in place.

        Returns
        -------
        Experiment
            experiment with reordered samples
        '''
        if inplace is False:
            exp = deepcopy(self)
        else:
            exp = self
        # make it a np array; otherwise the slicing won't work if the new_order is
        # a list of boolean and data is sparse matrix. For example:
        # from scipy.sparse import csr_matrix
        # a = csr_matrix((3, 4), dtype=np.int8)
        # In [125]: a[[False, False, False], :]
        # Out[125]:
        # <3x4 sparse matrix of type '<class 'numpy.int8'>'

        # In [126]: a[np.array([False, False, False]), :]
        # Out[126]:
        # <0x4 sparse matrix of type '<class 'numpy.int8'>'
        new_order = np.array(new_order)
        if axis == 0:
            exp.data = exp.data[new_order, :]
            exp.sample_metadata = exp.sample_metadata.iloc[new_order, :]
        elif axis == 1:
            exp.data = exp.data[:, new_order]
            exp.feature_metadata = exp.feature_metadata.iloc[new_order, :]

        return exp


def add_functions(cls,
                  modules=['.io', '.sorting', '.filtering',
                           '.transforming', '.heatmap.heatmap']):
    '''Dynamically add functions to the class as methods.

    Parameters
    ----------
    cls : ``class`` object
        The class that the functions will be added to
    modules : iterable of str
        The modules where the functions are defined
    '''
    for module_name in modules:
        module = import_module(module_name, 'calour')
        functions = inspect.getmembers(module, inspect.isfunction)
        # import ipdb; ipdb.set_trace()
        for fn, f in functions:
            # skip private functions
            if not fn.startswith('_'):
                setattr(cls, fn, f)


def join_experiments(exp, other, orig_field_name='orig_exp', orig_field_values=None, prefixes=None):
    '''Join two Experiment objects into one.

    If suffix is not none, add suffix to each sampleid (suffix is a
    list of 2 values i.e. ('_1','_2')) if same feature id in both
    studies, use values, otherwise put 0 in values of experiment where
    the observation in not present

    Parameters
    ----------
    exp, other : Experiments to join
    '''
    logger.debug('Join experiments:\n{!r}\n{!r}'.format(exp, other))
    newexp = deepcopy(exp)
    newexp.description = 'join %s & %s' % (exp.description, other.description)

    # test if we need to force a suffix (when both experiments contain the same sample ids)
    if len(exp.sample_metadata.index.intersection(other.sample_metadata.index)) > 0:
        if prefixes is None:
            raise ValueError('You need provide prefix to add to sample ids '
                             'because the two experiments has some same sample ids.')
        else:
            exp_prefix, other_prefix = prefixes
            logger.info('both experiments contain same sample id')
            exp_sample_metadata = exp.sample_metadata.copy()
            other_sample_metadata = other.sample_metadata.copy()
            exp_sample_metadata.index = ['{}_{!s}'.format(exp_prefix, i)
                                         for i in exp_sample_metadata.index]
            other_sample_metadata.index = ['{}_{!s}'.format(other_prefix, i)
                                           for i in other_sample_metadata.index]
    else:
        exp_sample_metadata = exp.sample_metadata
        other_sample_metadata = other.sample_metadata

    sample_metadata = pd.concat([exp_sample_metadata, other_sample_metadata], join='outer', )
    if orig_field_name is not None:
        sample_metadata[orig_field_name] = np.nan
        sample_metadata.loc[exp_sample_metadata.index.values, orig_field_name] = exp.description
        sample_metadata.loc[other_sample_metadata.index.values, orig_field_name] = other.description
    newexp.sample_metadata = sample_metadata

    sample_pos_exp = [sample_metadata.index.get_loc(csamp) for csamp in exp_sample_metadata.index.values]
    sample_pos_other = [sample_metadata.index.get_loc(csamp) for csamp in other_sample_metadata.index.values]

    feature_metadata = exp.feature_metadata.merge(other.feature_metadata, how='outer', left_index=True, right_index=True, suffixes=('', '__tmp_other'))
    # merge and remove duplicate columns
    keep_cols = []
    for ccol in feature_metadata.columns:
        if ccol.endswith('__tmp_other'):
            expcol = ccol[:-len('__tmp_other')]
            feature_metadata[expcol].fillna(feature_metadata[ccol], inplace=True)
        else:
            keep_cols.append(ccol)
    feature_metadata = feature_metadata[keep_cols]
    newexp.feature_metadata = feature_metadata

    all_features = feature_metadata.index.values
    all_data = np.zeros([len(sample_metadata), len(all_features)])
    data_exp = exp.get_data(sparse=False)
    data_other = other.get_data(sparse=False)
    logger.warn('data')

    for idx, cfeature in enumerate(all_features):
        if cfeature in exp.feature_metadata.index:
            all_data[sample_pos_exp, idx] = data_exp[:, exp.feature_metadata.index.get_loc(cfeature)]
        if cfeature in other.feature_metadata.index:
            all_data[sample_pos_other, idx] = data_other[:, other.feature_metadata.index.get_loc(cfeature)]
    newexp.data = all_data
    return newexp


def join_fields(exp, field1, field2, newfield):
    '''
    create a new sample metadata field by concatenating the values in the two fields specified
    '''


def merge_obs_tax(exp, tax_level=3, method='sum'):
    '''
    merge all observations with identical taxonomy (at level tax_level) by summing the values per sample
    '''


def _collapse_obs(exp, groups, method='sum'):
    '''
    collapse the observations based on values in groups (list of lists)
    '''


def merge_samples(exp, field, method='mean'):
    '''
    merge all samples that have the same value in field
    methods for merge (value for each observation) are:
    'mean' : the mean of all samples
    'random' : a random sample out of the group (same sample for all observations)
    'sum' : the sum of values in all the samples
    '''


def add_observation(exp, obs_id, data=None):
    '''
    add an observation to the experiment. fill the data with 0 if values is none, or with the values of data
    '''
