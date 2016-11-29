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
from calour.util import _get_taxonomy_string

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
    # print(metadata)
    if metadata is None:
        logger.info('No metadata associated with features in biom table')
    else:
        metadata = [dict(tmd) for tmd in metadata]
    md_df = pd.DataFrame(metadata, index=ids)
    # md_df['sequence']=ids
    md_df.index.name = 'sequence'
    return md_df


def _read_table(f):
    '''Read tab-delimited table file.

    It is used to read sample metadata (mapping) file and feature
    metadata file

    '''
    table = pd.read_table(f, sep='\t', index_col=0)
    # make sure the sample ID is string-type
    table.index = table.index.astype(np.str)
    return table


def read_bacteria(*kargs, **kwargs):
    exp = read(*kargs, **kwargs)
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
    type : str (optional)
    sparse : bool
        read the biom table into sparse or dense array
    '''
    logger.info('Reading experiment (biom table %s, map file %s)' % (data_file, sample_metadata_file))
    sid, oid, data, md = _read_biom(data_file, sparse=sparse)
    if sample_metadata_file is not None:
        # reorder the sample id to align with biom
        sample_metadata = _read_table(sample_metadata_file).loc[sid, ]
    else:
        sample_metadata = pd.DataFrame(index=sid)
    if feature_metadata_file is not None:
        # reorder the feature id to align with that from biom table
        fm = _read_table(feature_metadata_file).loc[oid, ]
        # combine it with the metadata from biom
        feature_metadata = pd.concat([fm, md], axis=1)
    else:
        feature_metadata = md
    return Experiment(data, sample_metadata, feature_metadata, description=description, sparse=sparse)


def serialize(exp, f):
    '''Serialize the Experiment object to disk.'''


def save(exp, prefix, fmt='hdf5'):
    '''Save the experiment data to disk.

    Parameters
    ----------
    prefix : str
        file path to save to.
    fmt : str
        'biom','txt' etc.
    '''
    exp.save_biom('%s.biom' % prefix, fmt=fmt)
    exp.save_sample_metadata('%s_sample.txt' % prefix)
    exp.save_feature_metadata('%s_feature.txt' % prefix)


def save_biom(exp, f, fmt='hdf5', addtax=True):
    '''Save experiment to biom format

    Parameters
    ----------
    f : str
        the f to save to
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
    '''save the sample metadata file '''
    exp.sample_metadata.to_csv(f, sep='\t')


def save_feature_metadata(exp, f):
    exp.feature_metadata.to_csv(f, sep='\t')


def save_commands(exp, f):
    '''
    save the commands used to generate the exp
    '''


def save_fasta(exp, f):
    '''
    '''


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
    table : biom_table
        the biom table representation of the experiment
    '''

    # init the table
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
