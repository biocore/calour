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
   join_experiments_featurewise
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
from collections import defaultdict

import pandas as pd
import numpy as np

from .experiment import Experiment
from .util import join_fields


logger = getLogger(__name__)


def chain(exp: Experiment, steps=[], inplace=False, **kwargs) -> Experiment:
    '''Perform multiple operations sequentially.

    Parameters
    ----------
    steps : list of callables
        Each callable is a class method that has a boolean
        parameter of ``inplace``, and returns an
        :class:`.Experiment` object.
    inplace : bool, default=False
        change occurs in place or not.
    kwargs : dict
        keyword arguments to pass to each class method. The dict
        key should be in the form of
        "<method_name>__<param_name>". For example,
        "exp.chain(steps=[filter_samples, log_n], log_n__n=3)"
        will call :func:`filter_samples` first and then
        :func:`log_n` while setting its parameter `n=3`.

    Returns
    -------
    Experiment

    '''
    exp = exp if inplace else deepcopy(exp)
    params = defaultdict(dict)
    for k, v in kwargs.items():
        transformer, param_name = k.split('__')
        if param_name == 'inplace':
            raise ValueError(
                'You can not set `inplace` for individual transformation.')
        params[transformer][param_name] = v
    for step in steps:
        step(exp, inplace=True, **params[step.__name__])
    return exp


def join_metadata_fields(exp: Experiment, field1, field2, new_field=None,
                         axis='s', inplace=True, **kwargs) -> Experiment:
    '''Join 2 fields in sample or feature metadata into 1.

    Parameters
    ----------
    field1 : str
        Name of the first field to join. The value in this column can be any data type.
    field2 : str
        Name of the field to join. The value in this column can be any data type.
    new_field : str, default=None
        name of the new (joined) field. Default to name it as field1 + sep + field2
    sep : str, optional
        The separator between the values of the two fields when joining
    kwargs : dict
        Other parameters passing to :func:`join_fields`.

    Returns
    -------
    Experiment

    See Also
    --------
    join_fields
    '''
    if not inplace:
        exp = deepcopy(exp)
    if axis == 0:
        md = exp.sample_metadata
    else:
        md = exp.feature_metadata

    join_fields(md, field1, field2, new_field, **kwargs)

    return exp


def aggregate_by_metadata(exp: Experiment, field, agg='mean', axis=0, inplace=False) -> Experiment:
    '''Aggregate all samples or features of the same group.

    Group the samples (axis=0) or features (axis=1) that have the same
    value in the column of given field and then aggregate the data
    table of each group with the given method.

    The number of samples/features in each group and their IDs are
    stored in new metadata columns '_calour_merge_number' and
    '_calour_merge_ids', respectively. For other metadata, the first
    one in the metadata table in each group is kept in the final
    returned experiment object.

    .. warning:: It will convert the ``Experiment.data`` from the
       sparse matrix to dense array.

    Parameters
    ----------
    field : str
        The sample/feature metadata field to group samples/features
    agg : str, optional
        aggregate method. Choice includes:

        * 'mean' : the mean of the group
        * 'median' : the median of the group
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
    if not inplace:
        exp = deepcopy(exp)
    if axis == 0:
        col = exp.sample_metadata[field]
    else:
        col = exp.feature_metadata[field]

    # convert to dense for efficient slicing
    exp.sparse = False

    uniq = col.unique()
    n = len(uniq)
    keep_pos = np.empty(n, dtype=np.uint32)
    merge_number = np.empty(n, dtype=np.uint32)
    # use object as dtype for string
    merge_ids = np.empty(n, dtype=object)

    for i, val in enumerate(uniq):
        if pd.isnull(val):
            pos = col.isnull()
        else:
            pos = col == val
        cdata = exp.data.compress(pos, axis=axis)
        if agg == 'mean':
            newdat = cdata.mean(axis=axis)
        elif agg == 'median':
            newdat = np.median(cdata, axis=axis)
        elif agg == 'sum':
            newdat = cdata.sum(axis=axis)
        else:
            raise ValueError('Unknown aggregation method: %r' % agg)
        merge_number[i] = pos.sum()
        merge_ids[i] = ';'.join(col.index[pos].astype(str))
        replace_pos = np.where(pos)[0][0]
        keep_pos[i] = replace_pos

        # if axis == 1, swap it to the first dimension to change column.
        exp.data.swapaxes(0, axis)[replace_pos] = newdat

    exp.reorder(keep_pos, axis=axis, inplace=True)

    if axis == 0:
        metadata = exp.sample_metadata
    else:
        metadata = exp.feature_metadata
    metadata['_calour_merge_number'] = merge_number
    metadata['_calour_merge_ids'] = merge_ids

    return exp


def join_experiments(exp: Experiment, other, field, labels=('exp', 'other'), prefixes=None) -> Experiment:
    '''Combine two :class:`.Experiment` objects into one.

    This assumes the same feature in the 2 joining experiments has
    the same ID. A new column will be added to the combined
    :attr:`.Experiment.sample_metadata` to store which of the 2
    combined objects every sample is from.

    Parameters
    ----------
    other : Experiment
        The ``Experiment`` object to combine with the current one. If
        both experiments contain the same feature metadata column and
        there is a conflict between the two, the value will be taken
        from exp and not from other.
    field : None or str
        Name of the new ``sample_metdata`` field containing the experiment each sample is coming from.
        If it is None, don't add such column. The values in this column will be "exp" and "other".
    labels : tuple of (str, str)
        Only used if `field` is not `None`. Label which experiments each sample is from.
    prefixes : tuple of (str, str), optional
        Prefix to prepend to the sample_metadata index for identical samples in the 2 experiments.
        Required only if the two experiments share any identical sample ID.

    Returns
    -------
    Experiment
        A new experiment with samples from both experiments concatenated, features from both
        experiments merged.

    '''
    logger.debug('Join 2 experiments:\n{!r}\n{!r}'.format(exp, other))
    # create an empty object of the same class type
    newexp = exp.__class__(np.empty(shape=[0, 0]), pd.DataFrame(),
                           description='join %s & %s' % (exp.description, other.description))

    if exp.normalized != other.normalized:
        raise ValueError(
            'Experiments are not normalized to same depth. Use exp.normalize() to normalize both first.')

    newexp.normalized = exp.normalized

    # merge sample metadata
    smd1 = exp.sample_metadata
    smd2 = other.sample_metadata
    smd = _check_id_overlap_then_concat(smd1, smd2, prefixes, field, labels)
    newexp.sample_metadata = smd

    # merge feature metadata
    suffix = '__OTHER__'
    fmd = pd.merge(exp.feature_metadata, other.feature_metadata,
                   how='outer', left_index=True, right_index=True,
                   suffixes=['', suffix])
    # combine and remove duplicate columns
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
    all_data[len(smd1):len(smd), idx] = other.get_data(sparse=False)
    newexp.data = all_data

    # validate the combined experiment at last
    newexp.validate()
    return newexp


def join_experiments_featurewise(exp: Experiment, other, field, labels=('exp', 'other'), prefixes=None) -> Experiment:
    '''Combine two :class:`.Experiment` objects into one.

    An example of use cases is to combine the 16S and ITS amplicon
    data together. If a sample only exist in one experiment, it will
    be dropped.

    .. warning:: If a sample has only features in one :class:`.Experiment`
       object and not the other, the sample will be dropped from joining.

    Parameters
    ----------
    other : :class:`.Experiment`
        The ``Experiment`` object to combine with the current one.
    field : ``None`` or str
        Name of the new ``feature_metdata`` field containing the experiment each feature is coming from.
        If it is None, don't add such column. The values in this column will be "exp" and "other".
    labels : tuple of (str, str)
        Only used if `field` is not `None`. Label which experiments each features is from.
    prefixes : tuple of (str, str), optional
        Prefix to prepend to the feature_metadata index for identical feature IDs in the 2 experiments.
        Required only if the two experiments share any identical feature ID.

    Returns
    -------
    :class:`.Experiment`
        A new experiment with features from both experiments concatenated

    '''
    logger.debug('Join 2 experiments featurewise:\n{!r}\n{!r}'.format(exp, other))
    # create an empty object
    newexp = exp.__class__(np.empty(shape=[0, 0]), pd.DataFrame(),
                           description='join %s & %s' % (exp.description, other.description))
    # intersect samples
    sid = exp.sample_metadata.index.intersection(other.sample_metadata.index)
    exp = exp.filter_ids(sid, axis=0)
    other = other.filter_ids(sid, axis=0)

    # merge features
    fmd1 = exp.feature_metadata
    fmd2 = other.feature_metadata

    newexp.feature_metadata = _check_id_overlap_then_concat(fmd1, fmd2, prefixes, field, labels)
    # assume exp and other have the same sample metadata because they
    # are the same set of samples.
    newexp.sample_metadata = exp.sample_metadata
    # merge data table
    newexp.data = np.c_[exp.data, other.data]

    return newexp


def _check_id_overlap_then_concat(df1, df2, prefixes, field, labels):
    intersect = df1.index.intersection(df2.index)
    if len(intersect) > 0:
        if prefixes is None:
            raise ValueError('You need provide the prefixes parameter to add to IDs, '
                             'because the two experiments have some identical '
                             'IDs:\n%r' % intersect)
        prefix1, prefix2 = prefixes
        logger.info('Both experiments contain same sample IDs - adding prefixes')
        if prefix1:
            df1.rename(lambda x: '{}_{!s}'.format(prefix1, x), inplace=True)
        if prefix2:
            df2.rename(lambda x: '{}_{!s}'.format(prefix2, x), inplace=True)

    df = pd.concat([df1, df2], join='outer')
    if field is not None:
        if field in df.columns:
            raise ValueError(
                'Column name %s already exists in the metadata - '
                'please give a different name' % field)
        df[field] = [labels[0]] * df1.shape[0] + [labels[1]] * df2.shape[0]
    return df
