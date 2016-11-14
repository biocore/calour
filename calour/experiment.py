# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from os.path import join
from copy import copy, deepcopy
from importlib import import_module
import inspect
from functools import wraps

import pandas as pd
import numpy as np
import scipy
import biom


logger = getLogger(__name__)


class Experiment:
    '''This class contains the data for a experiment or a meta experiment.

    The data set includes a data table (otu table, gene table,
    metabolomic table, or all those tables combined), a sample
    metadata table, and a feature metadata.

    Attributes
    ----------
    data : ``numpy.array`` or ``scipy.sparse``

    sample_metadata : ``pandas.DataFrame``

    feature_metadata : ``pandas.DataFrame``
    '''
    def __init__(self, data, sample_metadata, feature_metadata=None,
                 description='', sparse=True):
        self.data = data
        self.sample_metadata = sample_metadata
        self.feature_metadata = feature_metadata
        self.description = description

        # the function calling history list
        self._call_history = []
        # whether to log to history
        self._log = True

        # flag if data array is sparse (True) or dense (False)
        self.sparse = sparse

    def __repr__(self):
        '''
        print the information about the experiment
        should have number of samples, observations, first 3 sequences and first 3 samples?
        '''
        return('Experiment %s with %d samples, %d features' % (self.description,self.data.shape[0],self.data.shape[1]))

    def __copy__(self):
        '''Create a copy of Experiment
        '''
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        '''Create a deep copy of Experiment
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
            if exp._log is True:
                args_1 = ', '.join(args)
                args_2 = ', '.join('%s=%r' % (k, v) in kwargs.items)
                exp._call_history.append(
                    '{0}({1}, {2})'.format(fn, args_1, args_2))
                exp._log = False
            out = func(*args, **kwargs)
            # set log status back
            exp._log = log
            return out

        return inner

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


def add_functions(cls, modules=['.io', '.sorting', '.filtering', '.utils', '.normalization']):
    '''Dynamically add functions to the class as methods.'''
    for module_name in modules:
        module = import_module(module_name, 'calour')
        functions = inspect.getmembers(module, inspect.isfunction)
        # import ipdb; ipdb.set_trace()
        for fn, f in functions:
            # skip private functions
            if not fn.startswith('_'):
                setattr(cls, fn, f)


add_functions(Experiment)



def join_experiments(exp1, exp2, orig_field_name='orig_exp', orig_field_values=None, suffixes=None):
    '''
    join two Experiments into one experiment
    if suffix is not none, add suffix to each sampleid (suffix is a list of 2 values i.e. ('_1','_2'))
    if same observation id in both studies, use values, otherwise put 0 in values of experiment where the observation in not present
    '''


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
