# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger

import numpy as np
import skbio

from .experiment import Experiment
from .util import _get_taxonomy_string


logger = getLogger(__name__)


class AmpliconExperiment(Experiment):
    def __repr__(self):
        '''Return a string representation of this object.'''
        return 'AmpliconExperiment %s with %d samples, %d features' % (
            self.description, self.data.shape[0], self.data.shape[1])

    def plot(self, databases=('dbbact',), feature_field='taxonomy', **kwargs):
        # plot the experiment using taxonmy field and dbbact database
        super().plot(feature_field=feature_field, databases=databases, **kwargs)

    def filter_taxonomy(exp, values, negate=False, inplace=False, substring=True):
        '''filter keeping only observations with taxonomy string matching taxonomy

        if substring=True, look for partial match instead of identity.
        Matching is case insensitive

        Parameters
        ----------
        values : str or list of str
            the taxonomy string/strings to filter (can be partial if substring is True)
        negate : bool (optional)
            False (default) to keep matching taxonomies, True to remove matching taxonomies
        inplace : bool (optional)
            do the filtering on the original ``Experiment`` object or a copied one.
        substring : bool (optional)
            True (default) to do partial (substring) matching for the taxonomy string,
            False to do exact matching

        Returns
        -------
        ``AmpliconExperiment``
            With only features with matching taxonomy
        '''
        if 'taxonomy' not in exp.feature_metadata.columns:
            logger.warn('No taxonomy field in experiment')
            return None

        if not isinstance(values, (list, tuple)):
            values = [values]

        taxstr = exp.feature_metadata['taxonomy'].str.lower()

        select = np.zeros(len(taxstr), dtype=bool)
        for cval in values:
            if substring:
                select += [cval.lower() in ctax for ctax in taxstr]
            else:
                select += [cval.lower() == ctax for ctax in taxstr]

        if negate is True:
            select = ~ select

        logger.warn('%s remaining' % np.sum(select))
        return exp.reorder(select, axis=1, inplace=inplace)

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

    @Experiment._record_sig
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
        taxonomy = _get_taxonomy_string(exp, remove_underscore=True)
        sort_pos = np.argsort(taxonomy, kind='mergesort')
        exp = exp.reorder(sort_pos, axis=1, inplace=inplace)
        return exp

    @Experiment._record_sig
    def filter_orig_reads(exp, minreads, **kwargs):
        '''Filter keeping only samples with >= minreads in the original reads column
        Note this function uses the _calour_original_abundance field rather than the current sum of sequences per sample.
        So if you start with a sample with 100 reads, normalizing and filtering with other functions with not change the original reads column
        (which will remain 100).
        If you want to filter based on current total reads, use ``filter_by_data()`` instead

        Parameters
        ----------
        minreads : numeric
            Keep only samples with >= minreads reads (when loaded - not affected by normalization)

        Returns
        -------
        ``AmpliconExperiment`` - with only samples with enough original reads
        '''
        origread_field = '_calour_original_abundance'
        if origread_field not in exp.sample_metadata.columns:
            raise ValueError('%s field not initialzed. Did you load the data with calour.read_taxa() ?' % origread_field)

        good_pos = (exp.sample_metadata[origread_field] >= minreads).values
        newexp = exp.reorder(good_pos, axis=0, **kwargs)
        return newexp

    def plot_sort(exp, field=None, **kwargs):
        '''Plot bacteria after sorting by field
        This is a convenience wrapper for plot()
        Note: if sample_field is in **kwargs, use it as labels after sorting using field

        Parameters
        ----------
        field : str or None (optional)
            The field to sort samples by before plotting
        '''
        if field is not None:
            newexp = exp.sort_samples(field)
        else:
            newexp = exp
        if 'sample_field' in kwargs:
            newexp.plot(feature_field='taxonomy', **kwargs)
        else:
            newexp.plot(sample_field=field, feature_field='taxonomy', **kwargs)
