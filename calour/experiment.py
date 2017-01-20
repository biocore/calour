# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import scipy.sparse
from copy import copy, deepcopy
from importlib import import_module
import inspect
from functools import wraps

import pandas as pd
import numpy as np
import scipy


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
        Text describe the experiment
    sparse : bool
        Is data array in sparse or dense matrix

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

    def __repr__(self):
        '''Return a string representation of this object.

        should have number of samples, observations, first 3 sequences and first 3 samples?
        '''
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
        '''Create a copy of Experiment'''
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        '''Create a deep copy of Experiment

        Parameters
        ----------
        memo :
        '''
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

    def get_data(self, sparse=None, getcopy=False):
        '''Get the data 2d array

        Get the data 2d array (each column is a feature, row is a sample)

        Parameters
        ----------
        sparse : None or bool (optional)
            None (default) to pass original data format (sparse or dense).
            True to get as sparse.
            False to get as dense
        getcopy : bool (optional)
            True (default) to get a copy of the data
            False to get the original data (for inplace)
        '''
        if sparse is None:
            if getcopy:
                return self.data.copy()
            else:
                return self.data
        elif sparse:
            if scipy.sparse.issparse(self.data):
                if getcopy:
                    return self.data.copy()
                else:
                    return self.data
            else:
                return scipy.sparse.csr_matrix(self.data)
        else:
            if scipy.sparse.issparse(self.data):
                return self.data.toarray()
            else:
                if getcopy:
                    return self.data.copy()
                else:
                    return self.data

    def get_num_samples(self):
        '''Get the number of samples in the experiment
        '''
        return self.get_data().shape[0]

    def get_num_features(self):
        '''Get the number of features in the experiment
        '''
        return self.get_data().shape[1]

    def reorder(self, new_order, axis=0, inplace=False):
        '''Reorder according to indices in the new order.

        Note that we can also drop samples in new order.

        Parameters
        ----------
        new_order : Iterable of int
            the order of new indices
        axis : 0 or 1
        inplace : bool
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
        if axis == 0:
            exp.data = exp.data[new_order, :]
            exp.sample_metadata = exp.sample_metadata.iloc[new_order, :]
        elif axis == 1:
            exp.data = exp.data[:, new_order]
            exp.feature_metadata = exp.feature_metadata.iloc[new_order, :]

        return exp


def add_functions(cls, modules=['.io', '.sorting', '.filtering', '.transforming', '.heatmap']):
    '''Dynamically add functions to the class as methods.'''
    for module_name in modules:
        module = import_module(module_name, 'calour')
        functions = inspect.getmembers(module, inspect.isfunction)
        # import ipdb; ipdb.set_trace()
        for fn, f in functions:
            # skip private functions
            if not fn.startswith('_'):
                setattr(cls, fn, f)


def join(exp, other, orig_field_name='orig_exp', orig_field_values=None, prefixes=None):
    '''Join two Experiment objects into one.

    If suffix is not none, add suffix to each sampleid (suffix is a
    list of 2 values i.e. ('_1','_2')) if same feature id in both
    studies, use values, otherwise put 0 in values of experiment where
    the observation in not present

    Parameters
    ----------
    exp, other : 2 objects to join
    '''
    logger.debug('Join experiments:\n{!r}\n{!r}'.format(exp, other))
    newexp = copy.deepcopy(exp)
    newexp.description = 'join %s & %s' % (exp.description, other.description)

    # test if we need to force a suffix (when both experiments contain the same sample ids)
    if len(exp.sample_metadata.index.intersection(other.sample_metadata.index)) > 0:
        if prefixes is None:
            raise ValueError('You need provide prefix to add to sample ids '
                             'because the two experiments has some same sample ids.')
        else:
            exp_prefix, other_prefix = prefixes
            logger.info('both experiments contain same sample id')
            exp.sample_metadata.index = ['{}_{!s}'.format(exp_prefix, i)
                                         for i in exp.sample_metadata.index]
            other.sample_metadata.index = ['{}_{!s}'.format(other_prefix, i)
                                           for i in other.sample_metadata.index]
    sample_metadata = pd.concat([exp.sample_metadata, other.sample_metadata])

    common_exp = exp.filter_by_metadata()
    common_features = exp.feature_metadata.index.intersection(other.feature_metadata.index)
    newexp.feature_metadata.join(other.feature_metadata)

    # all_feature_md = list(set(exp1.feature_metadata.columns).union(set(exp2.feature_metadata.columns)))
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
