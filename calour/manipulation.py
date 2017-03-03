'''
analysis (:mod:`calour.manipulation`)
=====================================

.. currentmodule:: calour.manipulation

Functions
^^^^^^^^^
.. autosummary::
   :toctree: _autosummary

   join_fields
   join_experiments
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from copy import deepcopy
from logging import getLogger

import pandas as pd
import numpy as np

from . import Experiment


logger = getLogger(__name__)


@Experiment._convert_axis_name
def join_fields(exp, field1, field2, newname=None, axis=0, separator='_', inplace=True):
    '''Join two sample metadata fields into a single new field

    Parameters
    ----------
    field1 : str
        Name of the first sample metadata field to join
    field2 : str
        Name of the second sample metadata field to join
    newname : str or None (optional)
        name of the new (joined) sample metadata field
        None (default) to name it as field1-field2
    axis : int
        0 (default) to modify sample metadata fields, 1 to modify feature metadata fields
    separator : str (optional)
        The separator between the values of the two fields when joining
    inplace : bool (optional)
        True (default) to add in current experiment, False to create a new Experiment

    Returns
    -------
    newexp : Experiment
        with an added sample metadata field
    '''
    logger.debug('joining fields %s and %s into %s' % (field1, field2, newname))
    if inplace:
        newexp = exp
    else:
        newexp = deepcopy(exp)

    if axis == 0:
        metadata = newexp.sample_metadata
    else:
        metadata = newexp.feature_metadata

    # validate the data
    if field1 not in metadata.columns:
        raise ValueError('field %s not in metadata' % field1)
    if field2 not in metadata.columns:
        raise ValueError('field %s not in metadata' % field2)

    # get the new column name
    if newname is None:
        newname = field1 + separator + field2

    if newname in metadata.columns:
        raise ValueError('new field name %s alreay in metadata. Please use different newname value' % newname)

    # add the new column
    newcol = [str(i) + separator + str(j) for i, j in zip(metadata[field1], metadata[field2])]
    if axis == 0:
        newexp.sample_metadata[newname] = newcol
    else:
        newexp.feature_metadata[newname] = newcol

    return newexp


def merge_obs_tax(exp, tax_level=3, method='sum'):
    '''
    merge all observations with identical taxonomy (at level tax_level) by summing the values per sample
    '''


def _collapse_obs(exp, groups, method='sum'):
    '''
    collapse the observations based on values in groups (list of lists)
    '''


def merge_identical(exp, field, method='mean', axis=0, inplace=False):
    '''Merge all samples/features (for axis =0 / 1 respectively) that have the same value in field
    Methods for merge (value for each observation) are:
    'mean' : the mean of all samples/features
    'random' : a random sample/feature out of the group (same sample/feature for all observations)
    'sum' : the sum of values in all the samples/features

    Parameters
    ----------
    field : str
        The sample/feature metadata field
    method : str (optional)
        'mean' : the mean of all samples/features
        'random' : a random sample/feature out of the group (same sample/feature for all observations)
        'sum' : the sum of values in all the samples/features
    axis : 0/1 (optional)
        0 (default) to merge samples, 1 to merge features
    inplace : bool (optional)
        False (default) to create new Experiment, True to perform inplace

    Returns
    -------
    newexp : ``Experiment``
    '''
    logger.debug('merge samples using field %s, method %s' % (field, method))
    if inplace:
        newexp = exp
    else:
        newexp = deepcopy(exp)
    if axis == 0:
        metadata = newexp.sample_metadata
    else:
        metadata = newexp.feature_metadata
    if field not in metadata:
        raise ValueError('field %s not in metadata' % field)

    # convert to dense for efficient slicing
    newexp.sparse = False
    data = newexp.get_data()
    keep_pos = []
    for cval in metadata[field].unique():
        # in case the sample had nan is the value
        if pd.isnull(cval):
            pos = (metadata[field].isnull()).values
        else:
            pos = (metadata[field] == cval).values
        if axis == 0:
            cdata = data[pos, :]
        else:
            cdata = data[:, pos]
        if method == 'mean':
            newdat = cdata.mean(axis=axis)
        elif method == 'sum':
            newdat = cdata.sum(axis=axis)
        elif method == 'random':
            random_pos = np.random.randint(np.sum(pos))
            newdat = cdata[random_pos]
        replace_pos = np.where(pos)[0][0]
        keep_pos.append(replace_pos)
        if axis == 0:
            newexp.data[replace_pos, :] = newdat
        else:
            newexp.data[:, replace_pos] = newdat
    newexp.reorder(keep_pos, inplace=True, axis=axis)
    return newexp


def add_observation(exp, obs_id, data=None):
    '''
    add an observation to the experiment. fill the data with 0 if values is none, or with the values of data
    '''


def join_experiments(exp, other, orig_field_name='orig_exp', prefixes=None):
    '''Join two Experiment objects into one.

    Parameters
    ----------
    exp, other : ``Experiment``
        The experiments to join.
        If both experiments contain the same feature metadata column, the value will be taken from
        exp and not from other.
    orig_field_name : str (optional)
        Name of the new ``sample_metdata`` field containing the experiment each sample is coming from
    prefixes : tuple of (str,str) (optional)
        Prefix to append to the sample_metadata index for identical samples in the 2 experiments.
        Required only if the two experiments share an identical sample name

    Returns
    -------
    ``Experiment``
        A new experiment with samples from both experiments concatenated, features from both
        experiments merged.
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
            if exp_prefix:
                exp_sample_metadata.rename_axis(lambda x: '{}_{!s}'.format(exp_prefix, x), inplace=True)
            if other_prefix:
                other_sample_metadata.rename_axis(lambda x: '{}_{!s}'.format(other_prefix, x), inplace=True)
    else:
        exp_sample_metadata = exp.sample_metadata
        other_sample_metadata = other.sample_metadata

    # concatenate the sample_metadata
    sample_metadata = pd.concat([exp_sample_metadata, other_sample_metadata], join='outer', )
    if orig_field_name is not None:
        sample_metadata[orig_field_name] = np.nan
        sample_metadata.loc[exp_sample_metadata.index.values, orig_field_name] = exp.description
        sample_metadata.loc[other_sample_metadata.index.values, orig_field_name] = other.description
    newexp.sample_metadata = sample_metadata

    # and store the positions of samples from each experiment in the new samples list
    sample_pos_exp = [sample_metadata.index.get_loc(csamp) for csamp in exp_sample_metadata.index.values]
    sample_pos_other = [sample_metadata.index.get_loc(csamp) for csamp in other_sample_metadata.index.values]

    # merge the features
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

    # join the data of the two experiments, putting 0 if feature is not in an experiment
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
