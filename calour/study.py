# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from os.path import join

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

    Parameters
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

        # the command history list
        self.commands = []

    def __repr__(self):
        '''
        print the information about the experiment
        should have number of samples, observations, first 3 sequences and first 3 samples?
        '''


def reorder_samples(exp, neworder, inplace=False):
    '''
    reroder the samples in the experiment according to indices in neworder
    note that we can also drop samples in neworder

    output:
    newexp : Experiment with reordered samples
    '''


def reorder_obs(exp, neworder, inplace=False):
    '''
    reroder the observations in the experiment according to indices in neworder
    note that we can also drop samples in neworder

    output:
    newexp : Experiment with reordered samples
    '''


def copy_exp(exp):
    '''
    create a new copy of Experiment
    '''


def add_history():
    '''
    the decorator to add the history of each command to the experiment
    (how do we do it?)
    '''


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


# populate the class functions
Experiment.filter_samples = ca.filter_samples
