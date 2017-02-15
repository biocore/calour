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


logger = getLogger(__name__)


def join_fields(exp, field1, field2, newname, separator='-', inplace=False):
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
    inplace : bool (optional)
        False (default) to create a new Experiment, True to add in current experiment
    separator : str (optional)
        The separator between the values of the two fields when joining

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

    # validate the data
    if field1 not in newexp.sample_metadata.columns:
        raise ValueError('field %s not in sample metadata' % field1)
    if field2 not in newexp.sample_metadata.columns:
        raise ValueError('field %s not in sample metadata' % field2)

    # get the new column name
    if newname is None:
        newname = '%s-%s' % (field1, field2)

    # add the new column
    newcol = exp.sample_metadata[field1].str.cat(exp.sample_metadata[field2].astype(str), sep=separator)
    newexp.sample_metadata[newname] = newcol

    return newexp


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
