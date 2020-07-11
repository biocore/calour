'''
amplicon experiment (:mod:`calour.amplicon_experiment`)
=======================================================

.. currentmodule:: calour.amplicon_experiment

Classes
^^^^^^^
.. autosummary::
   :toctree: generated

   AmpliconExperiment
'''

# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from copy import deepcopy

import numpy as np
import matplotlib as mpl

from .experiment import Experiment
from .util import _get_taxonomy_string, _to_list


logger = getLogger(__name__)


class AmpliconExperiment(Experiment):
    '''This class stores amplicon data and associated metadata.

    This is a child class of :class:`.Experiment`.

    Parameters
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The abundance table for OTUs or ASVs. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    description : str
        name of experiment
    sparse : bool
        store the data array in :class:`scipy.sparse.csr_matrix`
        or :class:`numpy.ndarray`
    databases: iterable of str, optional
        database interface names to show by default in heatmap() function
        by default use 'dbbact'

    Attributes
    ----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        The abundance table for OTUs or ASVs. Samples
        are in row and features in column
    sample_metadata : pandas.DataFrame
        The metadata on the samples
    feature_metadata : pandas.DataFrame
        The metadata on the features
    shape : tuple of (int, int)
        the dimension of data
    sparse : bool
        store the data as sparse matrix (scipy.sparse.csr_matrix) or dense numpy array.
    info : dict
        information about the experiment (data md5, filenames, etc.)
    description : str
        name of the experiment
    databases : dict
        keys are the database names (i.e. 'dbbact' / 'gnps')
        values are the database specific data for the experiment (i.e. annotations for dbbact)

    See Also
    --------
    Experiment
    '''
    def __init__(self, *args, databases=('dbbact',), **kwargs):
        super().__init__(*args, databases=databases, **kwargs)

    def heatmap(self, *args, **kwargs):
        '''Plot a heatmap for the amplicon experiment.

        This method accepts exactly the same parameters as input with
        its parent class method and does exactly the sample plotting.

        The only difference is that by default, its color scale is **in
        log** as its `norm` parameter is set to
        `matplotlib.colors.LogNorm()`. It makes more sense to show the
        microbial abundances in color of log scale because they grow
        exponentially. You can always set it to other scale as
        explained in :meth:`.Experiment.heatmap`.

        See Also
        --------
        Experiment.heatmap

        '''
        # set this default value inside the function instead of on the
        # function API (like the __init__) because we don't wanna to
        # define mpl.colors.LogNorm() on the API; otherwise, vmin and
        # vmax are set the same once for all AmpliconExperiment
        # objects (which we don't want) because python initializes
        # the function arguments when it reads in its definition.
        if 'norm' not in kwargs:
            kwargs['norm'] = mpl.colors.LogNorm()
        super().heatmap(*args, **kwargs)

    def filter_by_taxonomy(self, values, field='taxonomy', substring=True, negate=False, inplace=False):
        '''Filter keeping only observations with taxonomy string matching taxonomy

        If substring=True, look for partial match instead of identity.
        Matching is case insensitive.

        Parameters
        ----------
        values : str or list of str
            the taxonomy string/strings to filter (can be partial if substring is True)
        negate : bool, optional
            False (default) to keep matching taxonomies, True to remove matching taxonomies
        inplace : bool, optional
            do the filtering on the original :class:`.Experiment` object or a copied one.
        substring : bool, optional
            True (default) to do partial (substring) matching for the taxonomy string,
            False to do exact matching.

        Returns
        -------
        AmpliconExperiment
            Containing only features with matching taxonomy
        '''
        values = _to_list(values)

        taxstr = self.feature_metadata[field].str.lower()

        select = np.zeros(len(taxstr), dtype=bool)
        for cval in values:
            if substring:
                select += [cval.lower() in ctax for ctax in taxstr]
            else:
                select += [cval.lower() == ctax for ctax in taxstr]

        if negate is True:
            select = ~ select

        logger.info('%s features remain.' % np.sum(select))
        return self.reorder(select, axis=1, inplace=inplace)

    def filter_by_fasta(self, fp, negate=False, inplace=False):
        '''Filter features from experiment based on fasta file

        Parameters
        ----------
        fp : str
            the fasta file path containing the sequences to use for filtering
        negate : bool, default=False
            False to keep only sequences matching the fasta file;
            True to remove sequences in the fasta file.
        inplace : bool, default=False
            False to create a copy of the experiment; True to filter inplace

        Returns
        -------
        newexp : Experiment
            filtered so contains only sequence present in exp and in the fasta file
        '''
        # put import here to avoid circular import
        from .io import _iter_fasta

        logger.debug('Filter by sequence using fasta file %s' % fp)
        okpos = []
        tot_seqs = 0

        for chead, cseq in _iter_fasta(fp):
            tot_seqs += 1
            cseq = cseq.upper()
            if cseq in self.feature_metadata.index:
                pos = self.feature_metadata.index.get_loc(cseq)
                okpos.append(pos)
        logger.debug('loaded %d sequences. found %d sequences in experiment' % (tot_seqs, len(okpos)))
        if negate:
            okpos = np.setdiff1d(np.arange(len(self.feature_metadata.index)), okpos, assume_unique=True)

        return self.reorder(okpos, axis=1, inplace=inplace)

    def sort_by_taxonomy(self, inplace=False):
        '''Sort the features based on the taxonomy.

        Sort features based on the taxonomy (alphabetical)

        Parameters
        ----------
        inplace : bool, optional
            False (default) to create a copy
            True to Replace data in exp

        Returns
        -------
        AmpliconExperiment
            sorted by taxonomy
        '''
        logger.debug('sort features by taxonomies')
        taxonomy = _get_taxonomy_string(self, remove_underscore=True)
        sort_pos = np.argsort(taxonomy, kind='mergesort')

        return self.reorder(sort_pos, axis=1, inplace=inplace)

    def filter_orig_reads(self, min_reads, inplace=False):
        '''Filter to keep only samples with >= min_reads in the original reads column.

        Note this function uses the `_calour_original_abundance` field
        rather than the current sum of abundance per sample. This
        field is auto created in sample_metadata when the experiment
        is loaded with :func:`read_amplicon`. If you want to filter
        based on current total reads, use :func:`filter_by_data()`
        instead.

        The purpose of this function is to remove samples with too few
        data for insufficient stat power.

        Parameters
        ----------
        min_reads : numeric
            Keep only samples with >= min_reads reads (not affected by normalization)

        Returns
        -------
        AmpliconExperiment
            with only samples with enough original reads

        '''
        field = '_calour_original_abundance'
        if field not in self.sample_metadata.columns:
            raise ValueError('%s field not initialzed. Did you load the data with calour.read_amplicon() ?' % field)

        good_pos = (self.sample_metadata[field] >= min_reads).values
        return self.reorder(good_pos, axis=0, inplace=inplace)

    def collapse_taxonomy(self, level='genus', inplace=False):
        '''Collapse all features sharing the same taxonomy up to level into a single feature

        Sums abundances of all features sharing the same taxonomy up to level.

        Parameters
        ----------
        level: str or int, optional
            the level to bin the taxonmies. can be int (0=kingdom, 1=phylum,...6=species)
            or a string ('kingdom' or 'k' etc.)
        inplace : bool, optional
            False (default) to create a copy
            True to Replace data in exp
        '''
        level_dict = {'kingdom': 0, 'k': 0,
                      'phylum': 1, 'p': 1,
                      'class': 2, 'c': 2,
                      'order': 3, 'o': 3,
                      'family': 4, 'f': 4,
                      'genus': 5, 'g': 5,
                      'species': 6, 's': 6}
        if not (isinstance(level, int) or level in level_dict):
            raise ValueError(
                'Unsupported taxonomy level %s. Please use out of %s' % (level, list(level_dict.keys())))
        level = level_dict.get(level, level)
        if inplace:
            newexp = self
        else:
            newexp = deepcopy(self)

        def _tax_level(tax_str, level):
            # local function to get taxonomy up to given level
            ctax = tax_str.split(';')
            level += 1
            if len(ctax) < level:
                ctax.extend(['other'] * (level - len(ctax)))
            return ';'.join(ctax[:level])

        newexp.feature_metadata['_calour_tax_group'] = newexp.feature_metadata['taxonomy'].apply(_tax_level, level=level)
        newexp.aggregate_by_metadata('_calour_tax_group', agg='sum', axis=1, inplace=True)
        newexp.feature_metadata['taxonomy'] = newexp.feature_metadata['_calour_tax_group']
        return newexp

    def split_taxonomy(self, field='taxonomy', sep=';',
                       names=['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']):
        '''Split taxonomy column into individual column per level.

        Parameters
        ----------
        sep : str
            the separator between taxa levels
        names : list
            the column names for the new columns split from ``field``

        Returns
        -------
        AmpliconExperiment

        Examples
        --------
        Assume the taxonomy string is in QIIME style:
        "k__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Staphylococcaceae;g__Staphylococcus;s__",
        You can split each taxonomy level to its own column by running:
        >>> exp.split_taxonomy()  #doctest: +SKIP
        '''
        self.feature_metadata[names] = self.feature_metadata[field].str.split(sep, expand=True)
        # return so you can chain the functions
        return self

    def get_lowest_taxonomy(self, sep=';', field='taxonomy', new_field='taxa'):
        '''Create a new column that contains the taxonomy of lowest possible level.

        For example, "k__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;
        f__Enterococcaceae;g__Enterococcus;s__" will return "g__Enterococcus".

        Parameters
        ----------
        sep : str
            the separator between taxa levels
        field : str
            column name that contains all levels of taxonomy
        new_field : str
            new column name

        Returns
        -------
        AmpliconExperiment

        '''
        def find_lowest(s, sep=sep):
            taxon = ''
            for i in s.split(sep):
                name = i.strip()
                if len(name) > 3:
                    taxon = name
                else:
                    return taxon
            return taxon

        self.feature_metadata[new_field] = self.feature_metadata[field].apply(find_lowest)
        return self
