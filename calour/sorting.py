# calour functions for sorting samples/observations
# functions should call reorder_samples() / reorder_obs()

from logging import getLogger

from copy import copy

import numpy as np
import scipy.sparse
from scipy import cluster, spatial
from sklearn.preprocessing import scale

import calour as ca
from calour import Experiment


logger = getLogger(__name__)


def sort_taxonomy(exp, inplace=False):
    '''Sort the features based on the taxonomy

    Sort features based on the taxonomy (alphabetical)

    Parameters
    ----------
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp
    Returns
    -------
    exp : Experiment
        sorted by taxonomy
    '''
    logger.debug('sorting by taxonomies')
    taxonomy = ca._get_taxonomy_string(exp, remove_underscore=True)
    sort_pos = np.argsort(taxonomy, kind='mergesort')
    exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
    return exp


@Experiment._record_sig
def cluster_features(exp, min_abundance=None, logit=True, log_cutoff=1, normalize=True, inplace=False):
    '''Cluster the features (similar features close to each other)
    Reorder the features so that ones with similar behavior (pattern across samples) are close to each other

    Parameters
    ----------
    min_abundance : None or float (optional)
        None (default) to not remove any features.
        float to remove all features with total reads < float (to make clustering faster).
    lgoit : bool (optional)
        True (default) to log transform the data before clustering.
        False to not log transform.
    log_cutoff : float (optional)
        if logit, this is the minimal read threshold for the log transform
    normalize : bool (optional)
        True (default) to normalize each feature to sum 1 std 1.
        False to not normalize each feature.
    inplace : bool (optional)
        False (default) to create a copy.
        True to Replace data in exp.
    Returns
    -------
    exp : Experiment
        With features filtered (if min_abundance is not None) and clsutered (reordered)
    '''
    # filter low-freq features
    if min_abundance is not None:
        logger.debug('filtering min abundance %d' % min_abundance)
        exp = exp.filter_by_data('sum_abundance', axis=1, inplace=inplace, cutoff=min_abundance)

    data = exp.get_data(sparse=False, getcopy=True)

    data = data.transpose()

    if logit:
        data[data < log_cutoff] = log_cutoff
        data = np.log2(data)

    if normalize:
        data = scale(data, axis=1, copy=False)

    # cluster
    dist_mat = spatial.distance.pdist(data, metric='euclidean')
    linkage = cluster.hierarchy.single(dist_mat)
    sort_order = cluster.hierarchy.leaves_list(linkage)

    exp = exp.reorder(sort_order, axis=1, inplace=inplace)
    return exp


def cluster_samples(exp, inplace=False):
    '''
    reorder samples by clustering similar samples
    '''


def sort_samples(exp, field, inplace=False):
    '''Sort samples based on sample metadata values in field

    Parameters
    ----------
    field : str
        Name of the field to sort by
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp
    Returns
    -------
    exp : Experiment
        With samples sorted by values in sample_metadata field
    '''
    logger.debug('sorting samples by field %s' % field)
    if field not in exp.sample_metadata.columns:
        raise ValueError('Field %s not found in sample metadata' % field)
    sort_pos = np.argsort(exp.sample_metadata[field], kind='mergesort')
    exp = exp.reorder(sort_pos, axis=0, inplace=inplace)
    return exp


def sort_freq(exp, logit=True, log_cutoff=1, sample_subset=None, inplace=False):
    '''Sort features based on their mean frequency
    Sort the features based on their mean (log) frequency (optional in a subgroup of samples).

    Parameters
    ----------
    logit : bool (optional)
        True (default) to calculate mean of the log2 transformed data (useful for reducing outlier effect)
        False to not log transform before mean calculation
    log_cutoff : float (optional)
        The minimal number of reads for the log trasnform (if logit=True)
    sample_subset : None or iterable of sample indices (optional)
        None (default) to sort based on mean in all samples in experiment
        (non-none) to sort based only on data from samples in the sample_subset
    inplace : bool (optional)
        False (default) to create a copy
        True to Replace data in exp
    Returns
    -------
    exp : Experiment
        With features sorted by mean frequency
    '''
    if sample_subset is None:
        sample_subset = exp
    else:
        if not sample_subset.feature_metadata.index.equals(exp.feature_metadata.index):
            raise ValueError('sample_subset features are different from sorting experiment features')

    if logit:
        data = sample_subset.get_data(sparse=False, getcopy=True)
        data[data < log_cutoff] = log_cutoff
        data = np.log2(data)
    else:
        data = sample_subset.get_data()

    sort_pos = np.argsort(data.mean(axis=0))
    exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
    return exp


def sort_obs_center_mass(exp, field=None, numeric=True, uselog=True, inplace=False):
    '''
    sort observations based on center of mass after sorting samples by field (or None not to pre sort)
    '''


def sort_seqs_first(exp, seqs, inplace=False):
    '''
    reorder observations by first putting the observations in seqs and then the others
    '''


def reverse_obs(exp, inplace=False):
    '''
    reverse the order of the observations
    '''


def sort_samples_by_seqs(exp, seqs, inplace=False):
    '''
    sort the samples based on the frequencies of sequences in seqs
    '''


def sort_niche(exp, field):
    '''
    sort by niches - jamie
    '''
