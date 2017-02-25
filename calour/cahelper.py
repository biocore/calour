# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


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

    newexp = exp.filter_min_abundance(min_reads)
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
