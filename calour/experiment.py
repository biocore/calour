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
        name of experiment
    sparse : bool
        store the data array in sparse (``scipy.sparse.csr_matrix``) matrix
        or numpy array

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
        store the data as sparse matrix (scipy.sparse.csr_matrix) or numpy array.
    description : str
        name of the experiment
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
        return scipy.sparse.issparse(self.data)

    @sparse.setter
    def sparse(self, sparse):
        if sparse is True and not scipy.sparse.issparse(self.data):
            self.data = scipy.sparse.csr_matrix(self.data)
        elif sparse is False and scipy.sparse.issparse(self.data):
            self.data = self.data.toarray()

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
                           '.transforming', '.heatmap.heatmap',
                           '.manipulation']):
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
