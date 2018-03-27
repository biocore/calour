'''
manipulation (:mod:`calour.manipulation`)
=========================================

.. currentmodule:: calour.manipulation

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   join_metadata_fields
   join_experiments
   aggregate_by_metadata
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

from .experiment import Experiment


logger = getLogger(__name__)


def join_metadata_fields(exp: Experiment, field1, field2, newfield=None, axis=0, sep='_', inplace=True):
    '''Join two sample/feature metadata fields into a single new field

    Parameters
    ----------
    field1 : str
        Name of the first sample metadata field to join
    field2 : str
        Name of the second sample metadata field to join
    newfield : str or None, optional
        name of the new (joined) sample metadata field
        None (default) to name it as field1 + sep + field2
    axis : 0, 1, 's', or 'f', optional
        0 or 's' (default) to modify sample metadata fields; 1 or 'f' to modify feature metadata fields
    sep : str, optional
        The separator between the values of the two fields when joining
    inplace : bool, optional
        True (default) to add in current experiment, False to create a new Experiment

    Returns
    -------
    Experiment
        with an added sample metadata field
    '''
    logger.debug('joining fields %s and %s into %s' % (field1, field2, newfield))
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
    if newfield is None:
        newfield = field1 + sep + field2

    if newfield in metadata.columns:
        raise ValueError('new field name %s alreay in metadata. Please use different newfield value' % newfield)

    # add the new column
    newcol = [str(i) + sep + str(j) for i, j in zip(metadata[field1], metadata[field2])]

    if axis == 0:
        newexp.sample_metadata[newfield] = newcol
    else:
        newexp.feature_metadata[newfield] = newcol

    return newexp


@Experiment._record_sig
def aggregate_by_metadata(exp: Experiment, field, agg='mean', axis=0, inplace=False):
    '''aggregate all samples/features that have the same value in the given field.

    Group the samples (axis=0) or features (axis=1) that have the same value in the column
    of given field and then aggregate the data table of the group with the given method.

    The number of samples/features in each group and their IDs are stored in new metadata columns
    '_calour_merge_number' and '_calour_merge_ids', respectively

    Parameters
    ----------
    field : str
        The sample/feature metadata field to group samples/features
    agg : str, optional
        aggregate method. Choice includes:

        * 'mean' : the mean of the group

        * 'random' : a random selection from the group

        * 'sum' : the sum of of the group
    axis : 0, 1, 's', or 'f', optional
        0 or 's' (default) to aggregate samples; 1 or 'f' to aggregate features
    inplace : bool, optional
        False (default) to create new Experiment, True to perform inplace

    Returns
    -------
    Experiment

    '''
    logger.debug('Merge data using field %s, agg %s' % (field, agg))
    if inplace:
        newexp = exp
    else:
        newexp = deepcopy(exp)
    if axis == 0:
        metadata = newexp.sample_metadata
    else:
        metadata = newexp.feature_metadata
    if field not in metadata:
        raise ValueError('Field %s not in metadata' % field)

    # convert to dense for efficient slicing
    newexp.sparse = False

    uniq = metadata[field].unique()
    n = len(uniq)
    keep_pos = np.empty(n, dtype=np.int16)
    merge_number = np.empty(n, dtype=np.int16)
    merge_ids = np.empty(n, dtype=object)

    for i, val in enumerate(uniq):
        # in case the metadata column has nan
        if pd.isnull(val):
            pos = metadata[field].isnull()
        else:
            pos = metadata[field] == val
        cdata = newexp.data.compress(pos, axis=axis)
        which_to_replace = 0
        if agg == 'mean':
            newdat = cdata.mean(axis=axis)
        elif agg == 'sum':
            newdat = cdata.sum(axis=axis)
        elif agg == 'random':
            random_pos = np.random.randint(np.sum(pos))
            newdat = cdata.take(random_pos, axis=axis)
            which_to_replace = random_pos
        else:
            raise ValueError('Unknown aggregation method: %r' % agg)
        merge_number[i] = pos.sum()
        merge_ids[i] = ';'.join(metadata.index[pos])
        replace_pos = np.where(pos)[0][which_to_replace]
        keep_pos[i] = replace_pos

        if axis == 0:
            newexp.data[replace_pos, :] = newdat
        else:
            newexp.data[:, replace_pos] = newdat

    newexp.reorder(keep_pos, inplace=True, axis=axis)

    if axis == 0:
        newexp.sample_metadata = newexp.sample_metadata.assign(_calour_merge_number=merge_number, _calour_merge_ids=merge_ids)
    else:
        newexp.feature_metadata = newexp.feature_metadata.assign(_calour_merge_number=merge_number, _calour_merge_ids=merge_ids)

    return newexp


@Experiment._record_sig
def join_experiments(exp: Experiment, other, field_name='experiments', prefixes=None):
    '''Combine two :class:`.Experiment` objects into one.

    A new column will be added to the combined
    :attr:`.Experiment.sample_metadata` to store which of the 2
    combined objects it is from.

    Parameters
    ----------
    other : Experiment
        The ``Experiment`` object to combine with the current one.  If
        both experiments contain the same feature metadata column and
        there is a conflict between the two, the value will be taken
        from exp and not from other.
    field_name : None or str, optional
        Name of the new ``sample_metdata`` field containing the experiment each sample is coming from.
        If it is None, don't add such column.
    prefixes : tuple of (str, str), optional
        Prefix to append to the sample_metadata index for identical samples in the 2 experiments.
        Required only if the two experiments share any identical sample ID

    Returns
    -------
    Experiment
        A new experiment with samples from both experiments concatenated, features from both
        experiments merged.

    '''
    logger.debug('Join 2 experiments:\n{!r}\n{!r}'.format(exp, other))
    # create an empty object
    newexp = exp.__class__(np.empty(shape=[0, 0]), pd.DataFrame(),
                           description='join %s & %s' % (exp.description, other.description))

    # merge sample metadata
    smd1 = exp.sample_metadata
    smd2 = other.sample_metadata
    # when both experiments contain the same sample ids)
    if len(exp.sample_metadata.index.intersection(other.sample_metadata.index)) > 0:
        if prefixes is None:
            raise ValueError('You need provide prefix to add to sample IDs '
                             'because the two experiments has some same sample IDs.')
        exp_prefix, other_prefix = prefixes
        logger.info('Both experiments contain same sample IDs - adding prefixes')
        if exp_prefix:
            smd1 = exp.sample_metadata.rename_axis(lambda x: '{}_{!s}'.format(exp_prefix, x), inplace=False)
        if other_prefix:
            smd2 = exp.sample_metadata.rename_axis(lambda x: '{}_{!s}'.format(other_prefix, x), inplace=False)
    # concatenate the sample_metadata
    smd = pd.concat([smd1, smd2], join='outer')
    if field_name is not None:
        smd[field_name] = np.nan
        smd.loc[smd1.index.values, field_name] = exp.description
        smd.loc[smd2.index.values, field_name] = other.description
    newexp.sample_metadata = smd

    # merge the feature metadata
    suffix = '__OTHER__'
    fmd = pd.merge(exp.feature_metadata, other.feature_metadata,
                   how='outer', left_index=True, right_index=True,
                   suffixes=['', suffix])
    # merge and remove duplicate columns
    keep_cols = []
    for ccol in fmd.columns:
        if ccol.endswith(suffix):
            expcol = ccol[:-len(suffix)]
            # for the NA cells, fill the column from exp with values from other
            fmd[expcol].fillna(fmd[ccol], inplace=True)
        else:
            keep_cols.append(ccol)
    newexp.feature_metadata = fmd[keep_cols]

    # merge data table
    # join the data of the two experiments, putting 0 if feature is not in an experiment
    all_features = fmd.index.tolist()
    all_data = np.zeros([len(smd), len(fmd)])
    idx = [all_features.index(i) for i in exp.feature_metadata.index]
    all_data[0:len(smd1), idx] = exp.get_data(sparse=False)
    idx = [all_features.index(i) for i in other.feature_metadata.index]
    all_data[len(smd1):(len(smd1) + len(smd2)), idx] = other.get_data(sparse=False)
    newexp.data = all_data

    return newexp


@Experiment._record_sig
def join_experiments_featurewise(exp: Experiment, other,
                                 field_name='_feature_origin_', origin_labels=('exp1', 'exp2')):
    '''Combine two :class:`.Experiment` objects into one.

    An example of user cases is to combine the 16S and ITS amplicon data together.

    .. warning:: If a sample has only features in one Experiment
    object and not the other, the sample will be dropped from joining.

    Parameters
    ----------
    other : :class:`.Experiment`
        The ``Experiment`` object to combine with the current one.  If
        both experiments contain the same feature metadata column and
        there is a conflict between the two, the value will be taken
        from exp and not from other.
    field_name : ``None`` or str (optional)
        Name of the new ``feature_metdata`` field containing the experiment each feature is coming from.
        If it is None, don't add such column.
    labels : tuple of (str, str) (optional)
        The text to label which experiment the feature is originated from.

    Returns
    -------
    :class:`.Experiment`
        A new experiment with samples from both experiments concatenated, features from both
        experiments merged.

    '''
    logger.debug('Join 2 experiments featurewise:\n{!r}\n{!r}'.format(exp, other))
    # create an empty object
    newexp = exp.__class__(np.empty(shape=[0, 0]), pd.DataFrame(),
                           description='join %s & %s' % (exp.description, other.description))
    sid = exp.sample_metadata.index.intersection(other.sample_metadata.index)
    exp = exp.filter_ids(sid, axis=0)
    other = other.filter_ids(sid, axis=0)
    fmd = pd.concat([exp.feature_metadata, other.feature_metadata], join='outer')
    fmd[field_name] = [origin_labels[0]] * exp.shape[1] + [origin_labels[1]] * other.shape[1]
    newexp.sample_metadata = exp.sample_metadata
    newexp.feature_metadata = fmd
    # merge data table
    newexp.data = np.c_[exp.data, other.data]

    return newexp
