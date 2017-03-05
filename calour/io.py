'''
read & write (:mod:`calour.io`)
===============================

.. currentmodule:: calour.io

Functions
^^^^^^^^^
.. autosummary::
   :toctree: generated

   read
   read_amplicon
   read_open_ms
   save
   save_biom
   save_sample_metadata
   save_feature_metadata
'''


# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import os.path

import pandas as pd
import biom

from .experiment import Experiment
from .amplicon_experiment import AmpliconExperiment
from .util import get_file_md5, get_data_md5, _get_taxonomy_string


logger = getLogger(__name__)


def _read_biom(fp, transpose=True):
    '''Read in a biom table file.

    Parameters
    ----------
    fp : str
        file path to the biom table
    transpose : bool (True by default)
        Transpose the table or not. The biom table has samples in
        column while sklearn and other packages require samples in
        row. So you should transpose the data table.

    Returns
    -------
    sid : list of str
        the sample ids
    oid : list of str
        the feature ids
    data : numpy array (2d) of float
        the table
    feature_md : :class:`pandas.DataFrame`
        the feature metadata (if availble in table)
    '''
    logger.debug('loading biom table %s' % fp)
    table = biom.load_table(fp)
    sid = table.ids(axis='sample')
    oid = table.ids(axis='observation')
    logger.info('loaded %d samples, %d observations' % (len(sid), len(oid)))
    data = table.matrix_data
    feature_md = _get_md_from_biom(table)

    if transpose:
        logger.debug('transposing table')
        data = data.transpose()

    return sid, oid, data, feature_md


def _get_md_from_biom(table):
    '''Get the metadata of last column in the biom table.

    Return
    ------
    ``pandas.DataFrame`` or ``None``
    '''
    ids = table.ids(axis='observation')
    metadata = table.metadata(axis='observation')
    if metadata is None:
        logger.info('No metadata associated with features in biom table')
        md_df = None
    else:
        md_df = pd.DataFrame([dict(tmd) for tmd in metadata], index=ids)
    return md_df


def _read_open_ms(fp, transpose=True):
    '''Read an OpenMS bucket table csv file

    Parameters
    ----------
    fp : str
        file path to the biom table
    transpose : bool (True by default)
        Transpose the table or not. The biom table has samples in
        column while sklearn and other packages require samples in
        row. So you should transpose the data table.

    Returns
    -------
    sid : list of str
        the sample ids
    oid : list of str
        the feature ids
    data : numpy array (2d) of float
        the table
    feature_md : pandas DataFram
        the feature metadata (if availble in table)
    '''
    logger.debug('loading OpenMS bucket table %s' % fp)
    # use the python engine as the default (c) engine throws an error
    # a known bug in pandas (see #11166)
    table = pd.read_csv(fp, header=0, engine='python')
    table.set_index(table.columns[0], drop=True, inplace=True)
    logger.info('loaded %d observations, %d  samples' % table.shape)
    sid = table.columns
    oid = table.index
    data = table.values.astype(float)
    if transpose:
        logger.debug('transposing table')
        data = data.transpose()
    return sid, oid, data


def _read_table(fp, encoding=None):
    '''Read tab-delimited table file.

    It is used to read sample metadata (mapping) file and feature
    metadata file

    Parameters
    ----------
    fp : str
        the file path to read
    encoding : str or None (optional)
        None (default) to use pandas default encoder, str to specify
        encoder name (see pandas.read_table() documentation)

    Returns
    -------
    pandas.DataFrame with index set to first column (as str)
    '''
    # read the 1st column as str
    table = pd.read_table(fp, sep='\t', encoding=encoding, dtype={0: str})
    # table.fillna('na', inplace=True)
    table.set_index(table.columns[0], drop=False, inplace=True)
    return table


def read_open_ms(data_file, sample_metadata_file=None, feature_metadata_file=None,
                 description=None, sparse=False, *, normalize, **kwargs):
    '''Load an OpenMS metabolomics experiment.

    Parameters
    ----------
    data_file : str
        name of the OpenMS bucket table CSV file
    sample_metadata_file : str or None (optional)
        None (default) to not load metadata per sample
        str to specify name of sample mapping file (tsv)
    feature_metadata_file : str or None (optional)
        Name of table containing additional metadata about each feature
        None (default) to not load
    description : str or None (optional)
        Name of the experiment (for display purposes).
        None (default) to assign file name
    sparse : bool (optional)
        False (default) to store data as dense matrix (faster but more memory)
        True to store as sparse (CSR)
    normalize : int or None
        normalize each sample to the specified reads. ``None`` to not normalize

    Returns
    -------
    exp : ``Experiment``
    '''
    logger.debug('Reading OpenMS data (OpenMS bucket table %s, map file %s)' % (data_file, sample_metadata_file))
    exp = read(data_file, sample_metadata_file, feature_metadata_file,
               data_file_type='openms', sparse=sparse,
               normalize=normalize,  **kwargs)

    exp.sample_metadata['id'] = exp.sample_metadata.index.values

    # generate nice M/Z (MZ) and retention time (RT) columns for each feature
    exp.feature_metadata['id'] = exp.feature_metadata.index.values
    mzdata = exp.feature_metadata['id'].str.split('_', expand=True)
    mzdata.columns = ['MZ', 'RT']
    mzdata = mzdata.astype(float)
    exp.feature_metadata = pd.concat([exp.feature_metadata, mzdata], axis='columns')

    return exp


def read(data_file, sample_metadata_file=None, feature_metadata_file=None,
         description='', sparse=True, data_file_type='biom', encoding=None,
         cls=Experiment, *, normalize):
    '''Read the files for the experiment.

    .. note:: The order in the sample and feature metadata tables are changed
       to align with biom table.

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
    file_type : str (optional)
        the data_file format. options:
        'biom' : a biom table (biom-format.org) (default)
        'openms' : an OpenMS bucket table csv (rows are feature, columns are samples)
    encoding : str or None (optional)
        encoder for the metadata files. None (default) to use
        pandas default encoder, str to specify encoder name (see
        pandas.read_table() documentation)
    cls : ``class``, optional
        what class object to read the data into (``Experiment`` by default)
    normalize : int or None
        normalize each sample to the specified reads. ``None`` to not normalize

    Returns
    -------
    data : np.array or scipy.sprase.csr
        The experiment count data (each row is a sample, each column is a feature)
    sample_metadata : :class:`pandas.DataFrame`
        Metadata for the samples
    feature_metadata : pandas.DataFrame
        Metadata for the features
    exp_metadata : dict
        information about the experiment (including 'map_md5', 'data_md5')
    description : str
        name of the experiment
    '''
    logger.debug('Reading experiment (%s, %s, %s)' % (
        data_file, sample_metadata_file, feature_metadata_file))
    exp_metadata = {'map_md5': ''}
    # load the data table
    if data_file_type == 'biom':
        sid, oid, data, md = _read_biom(data_file)
    elif data_file_type == 'openms':
        sid, oid, data = _read_open_ms(data_file)
    else:
        raise ValueError('unkown data_file_type %s' % data_file_type)
    # load the sample metadata file
    if sample_metadata_file is not None:
        # reorder the sample id to align with biom
        sample_metadata = _read_table(sample_metadata_file, encoding=encoding).loc[sid, ]
        exp_metadata['map_md5'] = get_file_md5(sample_metadata_file, encoding=encoding)
    else:
        sample_metadata = pd.DataFrame(index=sid)
        # duplicate the index to a column so we can select it
        sample_metadata['id'] = sid

    # load the feature metadata file
    if feature_metadata_file is not None:
        # reorder the feature id to align with that from biom table
        feature_metadata = _read_table(feature_metadata_file, encoding=encoding).loc[oid, ]
    else:
        feature_metadata = pd.DataFrame(index=oid)
        feature_metadata['id'] = oid
    if data_file_type == 'biom' and md is not None:
        # combine it with the metadata
        feature_metadata = pd.concat([feature_metadata, md], axis=1)

    # init the experiment metadata details
    exp_metadata['data_file'] = data_file
    exp_metadata['sample_metadata_file'] = sample_metadata_file
    exp_metadata['feature_metadata_file'] = feature_metadata_file
    exp_metadata['data_md5'] = get_data_md5(data)

    if description == '':
        description = os.path.basename(data_file)

    exp = cls(data, sample_metadata, feature_metadata,
              exp_metadata=exp_metadata, description=description, sparse=sparse)

    if normalize is not None:
        # record the original total read count into sample metadata
        exp.normalize(total=normalize, inplace=True)

    return exp


def read_amplicon(data_file, sample_metadata_file=None,
                  *, filter_reads, normalize, **kwargs):
    '''Load an amplicon experiment.

    Fix taxonomy, normalize reads, and filter low abundance
    samples. This wraps ``read()``.  Also convert feature metadata
    index (sequences) to upper case

    Parameters
    ----------
    sample_metadata_file : None or str (optional)
        None (default) to just use samplenames (no additional metadata).
    filter_reads : int or None
        int (default) to remove all samples with less than ``filter_reads``.
        ``None`` to not filter
    normalize : int or None
        normalize each sample to the specified reads. ``None`` to not normalize

    Returns
    -------
    ``AmpliconExperiment``
        after removing low read sampls and normalizing
    '''
    # don't do normalize before the possible filtering
    exp = read(data_file, sample_metadata_file, cls=AmpliconExperiment,
               normalize=None, **kwargs)

    exp.feature_metadata.index = exp.feature_metadata.index.str.upper()

    if 'taxonomy' in exp.feature_metadata.columns:
        exp.feature_metadata['taxonomy'] = _get_taxonomy_string(exp)
    else:
        exp.feature_metadata['taxonomy'] = 'NA'

    if filter_reads is not None:
        exp.filter_by_data('sum_abundance', cutoff=filter_reads, inplace=True)
    if normalize is not None:
        exp.normalize(total=normalize, inplace=True)

    return exp


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


def save_biom(exp, f, fmt='hdf5', add_metadata='taxonomy'):
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
    add_metadata : str or None (optional)
        add metadata column from ``Experiment.feature_metadata`` to biom table.
        Don't add if it is ``None``.

    '''
    logger.debug('save biom table to file %s format %s' % (f, fmt))
    tab = _create_biom_table_from_exp(exp, add_metadata)
    if fmt == 'hdf5':
        with biom.util.biom_open(f, 'w') as f:
            tab.to_hdf5(f, "calour")
    elif fmt == 'json':
        with open(f, 'w') as f:
            tab.to_json("calour", f)
    elif fmt == 'txt':
        if add_metadata:
            logger.warning('.txt format does not support taxonomy information in save. Saving without taxonomy.')
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


def save_fasta(exp, f, seqs=None):
    '''Save a list of sequences to fasta.
    Use taxonomy information if available, otherwise just use sequence as header.

    Parameters
    ----------
    f : str
        the filename to save to
    seqs : list of str sequences ('ACGT') or None (optional)
        None (default) to save all sequences in exp, or list of sequences to only save these sequences.
        Note: sequences not in exp will not be saved
    '''
    logger.debug('save_fasta to file %s' % f)
    if seqs is None:
        logger.debug('no sequences supplied - saving all sequences')
        seqs = exp.feature_metadata.index.values
    num_skipped = 0
    if 'taxonomy' in exp.feature_metadata.columns:
        add_taxonomy = True
    else:
        logger.debug('no taxonomy field in experiment. saving without taxonomy')
        add_taxonomy = False
    with open(f, 'w') as fasta_file:
        for idx, cseq in enumerate(seqs):
            if cseq not in exp.feature_metadata.index:
                num_skipped += 1
                continue
            if add_taxonomy:
                cheader = '%d %s' % (idx, exp.feature_metadata['taxonomy'][cseq])
            else:
                cheader = '%d %s' % (idx, cseq)
            fasta_file.write('>%s\n%s\n' % (cheader, cseq))
    logger.debug('wrote fasta file with %d sequences. %d sequences skipped' % (len(seqs)-num_skipped, num_skipped))


def _create_biom_table_from_exp(exp, add_metadata='taxonomy'):
    '''Create a biom table from an experiment

    Parameters
    ----------
    exp : Experiment
    add_metadata : str or None (optional)
        add metadata column from ``Experiment.feature_metadata`` to biom table.
        Don't add if it is ``None``.

    Returns
    -------
    biom_table
        the biom table representation of the experiment
    '''
    features = exp.feature_metadata.index
    samples = exp.sample_metadata.index
    table = biom.table.Table(exp.data.transpose(), features, samples, type="OTU table")
    # and add metabolite name as taxonomy:
    if add_metadata is not None:
        # md has to be a dict of dict, so it needs to be converted from
        # a DataFrame instead of Series
        md = exp.feature_metadata.loc[:, [add_metadata]].to_dict('index')
        table.add_metadata(md, axis='observation')
    return table
