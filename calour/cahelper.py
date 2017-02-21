# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import copy
from logging import getLogger
import skbio

logger = getLogger(__name__)


def filter_taxonomy(exp, taxonomy, exact=False, negate=False, **kwargs):
    '''Filter keeping only specif taxonomies

    Parameters
    ----------
    taxonomy : str or list of str
        the taxonomies names to filter
    exact : bool (optional)
        False (default) to allow substring match, True to require full match
    negate : bool (optional)
        False (default) to return matching features, True to return non-matching features

    Returns
    -------
    newexp : Experiment
    '''
    if 'taxonomy' not in exp.feature_metadata.columns:
        raise ValueError('taxonomy field not initialzed. Did you load the data with calour.read_taxa() ?')
    if not isinstance(taxonomy, (list, tuple)):
        taxonomy = [taxonomy]

    if exact:
        newexp = exp.filter_by_metadata('taxonomy', taxonomy, axis=1, negate=negate, **kwargs)
        return newexp

    select = exp.feature_metadata['taxonomy'] is None
    for ctax in taxonomy:
        select = select | exp.feature_metadata['taxonomy'].str.contains(ctax, case=False)
    if negate:
        select = ~ select
    newexp = exp.reorder(select.values, axis=1, **kwargs)
    return newexp


def cluster_features(exp, minreads=10, **kwargs):
    '''Cluster features following log transform and filtering of minimal reads
    '''
    if minreads > 0:
        newexp = filter_min_reads(exp, minreads)
    else:
        newexp = exp
    newexp = newexp.cluster_data(transform=log_and_scale, axis=0, **kwargs)
    return newexp


def sort_samples(exp, field, **kwargs):
    '''Sort experiment by field
    '''
    newexp = exp.sort_by_metadata(field=field, **kwargs)
    return newexp


def plot_s(exp, field=None, **kwargs):
    '''Plot bacteria (with taxonomy) after sorting by field
    use after load_taxa()
    note: if sample_field is in **kwargs, use it as labels after sorting using field
    '''
    newexp = sort_samples(exp, field)
    if 'sample_field' in kwargs:
        newexp.plot(feature_field='taxonomy', max_features=100, **kwargs)
    else:
        newexp.plot(field, feature_field='taxonomy', max_features=100, **kwargs)


def filter_min_reads(exp, minreads, **kwargs):
    '''filter keeping only features with >= minreads total
    '''
    newexp = exp.filter_by_data('sum_abundance', axis=1, cutoff=minreads, **kwargs)
    return newexp


def filter_orig_reads(exp, minreads, **kwargs):
    ''' filter keeping only samples with >= minreads
    '''
    origread_field = '_calour_original_abundance'
    if origread_field not in exp.sample_metadata.columns:
        raise ValueError('%s field not initialzed. Did you load the data with calour.read_taxa() ?' % origread_field)

    good_pos = (exp.sample_metadata[origread_field] >= minreads).values
    newexp = exp.reorder(good_pos, axis=0, **kwargs)
    return newexp


def filter_prevalence(exp, fraction=0.5, cutoff=1/10000, **kwargs):
    ''' filter sequences present in at least fraction fraction of the samples.

    Parameters
    ----------
    fraction : float (optional)
        Keep features present at least in fraction of samples
    cutoff : float (optional)
        The minimal fraction of reads for the otu to be called present in a sample

    Returns
    -------
    ``Experiment`` with only features present in at least fraction of samples
    '''
    newexp = exp.filter_by_data('prevalence', axis=1, fraction=fraction, cutoff=cutoff, **kwargs)
    return newexp


def filter_mean(exp, cutoff=0.01, **kwargs):
    ''' filter sequences with a mean at least cutoff
    '''
    factor = np.mean(exp.data.sum(axis=1))
    newexp = exp.filter_by_data('mean_abundance', axis=1, cutoff=cutoff * factor, **kwargs)
    return newexp


def filter_feature_ids(exp, ids, negate=False, inplace=False):
    '''Filter features based on a list of feature ids (index values)

    Parameters
    ----------
    ids : iterable of str
        the feature ids to filter
    negate : bool (optional)
        False (default) to keep only sequences matching the fasta file, True to remove sequences in the fasta file.
    inplace : bool (optional)
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    newexp : Experiment
        filtered so contains only sequence present in exp and in the fasta file
    '''
    logger.debug('filter_feature_ids for %d features input' % len(ids))
    okpos = []
    tot_ids = 0
    for cid in ids:
        tot_ids += 1
        if cid in exp.feature_metadata.index:
            pos = exp.feature_metadata.index.get_loc(cid)
            okpos.append(pos)
    logger.debug('loaded %d sequences. found %d sequences in experiment' % (tot_ids, len(okpos)))
    if negate:
        okpos = np.setdiff1d(np.arange(len(exp.feature_metadata.index)), okpos, assume_unique=True)

    newexp = exp.reorder(okpos, axis=1, inplace=inplace)
    return newexp


def filter_fasta(exp, filename, negate=False, inplace=False):
    '''Filter features from experiment based on fasta file

    Parameters
    ----------
    filename : str
        the fasta filename containing the sequences to use for filtering
    negate : bool (optional)
        False (default) to keep only sequences matching the fasta file, True to remove sequences in the fasta file.
    inplace : bool (optional)
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    newexp : Experiment
        filtered so contains only sequence present in exp and in the fasta file
    '''
    logger.debug('filter_fasta using file %s' % filename)
    okpos = []
    tot_seqs = 0
    for cseq in skbio.read(filename, format='fasta'):
        tot_seqs += 1
        cseq = str(cseq).upper()
        if cseq in exp.feature_metadata.index:
            pos = exp.feature_metadata.index.get_loc(cseq)
            okpos.append(pos)
    logger.debug('loaded %d sequences. found %d sequences in experiment' % (tot_seqs, len(okpos)))
    if negate:
        okpos = np.setdiff1d(np.arange(len(exp.feature_metadata.index)), okpos, assume_unique=True)

    newexp = exp.reorder(okpos, axis=1, inplace=inplace)
    return newexp


def sort_freq(exp, field=None, value=None, inplace=False):
    '''Sort features based on their abundance in a subset of the samples

    Parameters
    ----------
    field : str or None (default)
        None (default) to sort on all samples, str to sort only on samples matching the field/value combination
    value : str or list of str or None (default)
        if field is not None, value is the value/list of values so sorting is only on samples matching this list
    inplace : bool (optional)
        False (default) to create a copy of the experiment, True to filter inplace

    Returns
    -------
    newexp : Experiment
        with features sorted by abindance
    '''
    if field is None:
        subset = None
    else:
        if not isinstance(value, (list, tuple)):
            value = [value]
        subset = np.where(exp.sample_metadata[field].isin(value).values)[0]

    newexp = exp.sort_by_data(axis=1, subset=subset, inplace=inplace)
    return newexp


def is_sample_v4(exp, region_seq='TACG', frac_have=0.4, min_reads=10):
    '''Test which samples in the experiment are not from the region.
    Based on the consensus sequence at the beginning of the region.

    Parameters
    ----------
    region : str (optional)
        The nucelotide sequence which is the consensus
    frac_have : float (optional)
        The fraction (per sample) of sequences containing the consensus in order to be from the region
    min_reads : float
        test only sequences with at least total min_reads

    Returns
    -------
    good_samples : list of str
        List of samples which have at least frac_have of sequences matching region_seq
    bad_samples : list of str
        List of samples which don't have at least frac_have of sequences matching region_seq
    '''

    newexp = filter_min_reads(exp, min_reads)
    seqs_ok = newexp.feature_metadata.index.str.startswith(region_seq)
    num_seqs_ok = np.sum(newexp.data[:, seqs_ok] > 0, axis=1)
    num_seqs = np.sum(newexp.data > 0, axis=1)
    frac_ok = num_seqs_ok / num_seqs
    ok_samples = np.where(frac_ok >= frac_have)[0]
    bad_samples = np.where(frac_ok < frac_have)[0]
    return list(newexp.sample_metadata.index[ok_samples]), list(newexp.sample_metadata.index[bad_samples])


def set_log_level(level):
    '''Set the debug level for calour

    Parameters
    ----------
    level : int
        10 for debug, 20 for info, 30 for warn, etc.
    '''

    clog = getLogger('calour')
    clog.setLevel(level)


def log_and_scale(exp):
    exp.log_n(inplace=True)
    exp.scale(inplace=True, axis=0)
    return exp
