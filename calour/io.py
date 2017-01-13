# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger

import pandas as pd
import numpy as np
import scipy
import biom

from calour.experiment import Experiment
from calour.util import _get_taxonomy_string, get_file_md5, get_data_md5

logger = getLogger(__name__)


def _read_biom(fp, transpose=True, sparse=True):
    '''Read in a biom table file.

    Parameters
    ----------
    fp : str
        file path to the biom table
    transpose : bool
        Transpose the table or not. The biom table has samples in
        column while sklearn and other packages require samples in
        row. So you should transpose the data table.
    '''
    logger.debug('loading biom table %s' % fp)
    table = biom.load_table(fp)
    sid = table.ids(axis='sample')
    oid = table.ids(axis='observation')
    logger.info('loaded %d samples, %d observations' % (len(sid), len(oid)))
    if sparse:
        logger.debug('storing as sparse matrix')
        data = scipy.sparse.csr_matrix(table.matrix_data)
    else:
        logger.debug('storing as dense matrix')
        data = table.matrix_data.toarray()

    feature_md = _get_md_from_biom(table)

    if transpose:
        logger.debug('transposing table')
        data = data.transpose()

    return sid, oid, data, feature_md


def _get_md_from_biom(table):
    '''Get the metadata of last column in the biom table.

    Return
    ------
    pandas.DataFrame
    '''
    ids = table.ids(axis='observation')
    metadata = table.metadata(axis='observation')

    if metadata is None:
        logger.info('No metadata associated with features in biom table')
    else:
        metadata = [dict(tmd) for tmd in metadata]
    md_df = pd.DataFrame(metadata)
    md_df['ids'] = ids
    md_df.set_index('ids', drop=False, inplace=True)
    # md_df.index.name = 'sequence'
    return md_df


def _read_table(f):
    '''Read tab-delimited table file.

    It is used to read sample metadata (mapping) file and feature
    metadata file

    '''
    table = pd.read_table(f, sep='\t')
    table.set_index(table.columns[0], drop=False, inplace=True)
    # make sure the sample ID is string-type
    table.index = table.index.astype(np.str)
    return table


def read_taxa(data_file, sample_metadata_file=None,
              filter_orig_reads=1000, normalize=True, **kwargs):
    '''Load an amplicon experiment.

    Fix taxonomy and normalize if needed. This is a convenience function of read()

    Parameters
    ----------
    filter_orig_reads : int or None (optional)
        int (default) to remove all samples with < filter_orig_reads total reads. None to not filter
    normalize : bool (optional)
        True (default) to normalize each sample to 10000 reads

    Returns
    -------
    exp : Experiment
    '''
    exp = read(data_file, sample_metadata_file, **kwargs)
    if 'taxonomy' in exp.feature_metadata.columns:
        exp.feature_metadata['taxonomy'] = _get_taxonomy_string(exp)

    if filter_orig_reads is not None:
        exp.filter_by_data('sum_abundance', cutoff=filter_orig_reads, inplace=True)
    if normalize:
        # record the original total read count into sample metadata
        exp.sample_metadata['_calour_read_count'] = exp.data.sum(axis=1)
        exp.normalize(inplace=True)
    return exp


def read(data_file, sample_metadata_file=None, feature_metadata_file=None,
         description='', sparse=True):
    '''Read the files for the experiment.

    Parameters
    ----------
    data_file : str
        file path to the biom table.
    sample_metadata_file : None or str (optional)
        None (default) to just use samplenames (no additional metadata).
        if not None, file path to the sample metadata (aka mapping file in QIIME).
    feature_metadata_file : str
        file path to the feature metadata.
    description : str
        description of the experiment
    sparse : bool
        read the biom table into sparse or dense array

    Returns
    -------
    exp : Experiment
    '''
    logger.info('Reading experiment (biom table %s, map file %s)' % (data_file, sample_metadata_file))
    exp_metadata = {'map_md5': ''}
    sid, oid, data, md = _read_biom(data_file, sparse=sparse)
    if sample_metadata_file is not None:
        # reorder the sample id to align with biom
        sample_metadata = _read_table(sample_metadata_file).loc[sid, ]
        exp_metadata['map_md5'] = get_file_md5(sample_metadata_file)
    else:
        sample_metadata = pd.DataFrame(index=sid)
    if feature_metadata_file is not None:
        # reorder the feature id to align with that from biom table
        fm = _read_table(feature_metadata_file).loc[oid, ]
        # combine it with the metadata from biom
        feature_metadata = pd.concat([fm, md], axis=1)
    else:
        feature_metadata = md

    # init the experiment metadata details
    exp_metadata['data_file'] = data_file
    exp_metadata['sample_metadata_file'] = sample_metadata_file
    exp_metadata['feature_metadata_file'] = feature_metadata_file
    exp_metadata['data_md5'] = get_data_md5(data)

    return Experiment(data, sample_metadata, feature_metadata,
                      exp_metadata=exp_metadata, description=description, sparse=sparse)


def serialize(exp, f):
    '''Serialize the Experiment object to disk.'''


def save(exp, prefix, fmt='hdf5'):
    '''Save the experiment data to disk.

    Parameters
    ----------
    prefix : str
        file path to save to.
    fmt : str
        format for the data table. could be 'hdf5', 'txt', or 'json'.
    '''
    exp.save_biom('%s.biom' % prefix, fmt=fmt)
    exp.save_sample_metadata('%s_sample.txt' % prefix)
    exp.save_feature_metadata('%s_feature.txt' % prefix)


def save_biom(exp, f, fmt='hdf5', addtax=True):
    '''Save experiment to biom format

    Parameters
    ----------
    f : str
        the file to save to
    fmt : str (optional)
        the output biom table format. options are:
        'hdf5' (default) save to hdf5 biom table.
        'json' same to json biom table.
        'txt' save to text (tsv) biom table.
    addtax : bool (optional)
        True (default) to save taxonomy of features.
        False to not save taxonomy
    '''
    logger.debug('save biom table to file %s format %s' % (f, fmt))
    tab = _create_biom_table_from_exp(exp, addtax=addtax)
    if fmt == 'hdf5':
        with biom.util.biom_open(f, 'w') as f:
            tab.to_hdf5(f, "calour")
    elif fmt == 'json':
        with open(f, 'w') as f:
            tab.to_json("calour", f)
    elif fmt == 'txt':
        s = tab.to_tsv()
        with open(f, 'w') as f:
            f.write(s)
    else:
        raise ValueError('Unknwon file format %s for save' % fmt)
    logger.debug('biom table saved to file %s' % f)


def save_sample_metadata(exp, f):
    '''Save sample metadata to file. '''
    exp.sample_metadata.to_csv(f, sep='\t')


def save_feature_metadata(exp, f):
    '''Save feature metadata to file. '''
    exp.feature_metadata.to_csv(f, sep='\t')


def save_commands(exp, f):
    '''Save the commands used to generate the exp '''


def save_fasta(exp, f):
    ''' '''


def _create_biom_table_from_exp(exp, addtax=True):
    '''Create a biom table from an experiment

    Parameters
    ----------
    input:
    expdat : Experiment
    addtax : bool (optional)
        True (default) to add taxonomy metadata.
        False to not add taxonomy

    Returns
    -------
    biom_table
        the biom table representation of the experiment
    '''
    features = exp.feature_metadata.index
    samples = exp.sample_metadata.index
    table = biom.table.Table(exp.data.transpose(), features, samples, type="OTU table")
    # and add metabolite name as taxonomy:
    if addtax:
        taxdict = exp.feature_metadata.T.to_dict()
        table.add_metadata(taxdict, axis='observation')
        # metadata = table.metadata(axis='observation')
        # print(metadata)
    return table
