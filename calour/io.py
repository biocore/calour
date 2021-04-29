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
   read_qiime2
   read_ms
   save
   save_fasta
   save_biom
'''


# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
import zipfile
import tempfile

import pandas as pd
import numpy as np
import biom

from . import Experiment, AmpliconExperiment, MS1Experiment
from .util import get_file_md5, get_data_md5, _get_taxonomy_string
from ._doc import ds
from .database import _get_database_class


logger = getLogger(__name__)


def _file_from_zip(tempdir, data_file, internal_data):
    '''extract the data file from a regular/qza filename into the tempdir, and return the filename

    Parameters
    ----------
    tmpdir: str
        name of the directory to extract the zip into
    data_file: str
        original name of the file (could be '.qza' or not)
    internale_data: str
        the internal qiime2 qza file name (i.e. 'data/feature-table.biom' for biom table etc.)

    Returns
    -------
    str: name of data file inside the tempdir
    '''
    if not zipfile.is_zipfile(data_file):
        raise ValueError('the qiime2 file %s is not a valid zip file. Is it a qiime2 artifact (.qza) file?' % data_file)
    with zipfile.ZipFile(data_file) as fl:
        internal_name = None
        for fname in fl.namelist():
            if fname.endswith(internal_data):
                internal_name = fname
                data_file = fl.extract(internal_name, tempdir)
                return data_file

        raise ValueError('No data file (%s) in qza file %s. is it the appropriate qiime2 file?' % (internal_data, data_file))


def _read_qiime2_zip(fp, sample_in_row=False):
    '''Read in a qiime2 qza biom table

    NOTE: this function reads2 the qiime2 qza artifacts files using unzip rather than relying on qiime2.
    This enables reading qiime2 artifacts without activating the qiime2 environment

    Parameters
    ----------
    fp : str
        file path to the qiime2 (.qza) biom table
    sample_in_row: bool, optional
        False (defailt) - samples correspond to columns
        if True, samples in the biom table are in rows

    Returns
    -------
    sid : list of str
        the sample ids
    fid : list of str
        the feature ids
    data : numpy array (2d) of float
        the table
    feature_md : pandas.DataFrame
        the feature metadata (if availble in table)
    '''
    logger.debug('loading qiime2 feature table %s' % fp)

    # load the feature table file
    with tempfile.TemporaryDirectory() as tempdir:
        oname = _file_from_zip(tempdir, fp, internal_data='data/feature-table.biom')
        sid, fid, data, fmd = _read_biom(oname, sample_in_row=sample_in_row)

    return sid, fid, data, fmd


def read_qiime2(fp, sample_metadata_file=None, rep_seq_file=None, taxonomy_file=None, **kwargs):
    '''Read a qiime2 artifact files into an amplicon experiment.


    .. note:: It converts feature metadata index (sequences) to
       upper case.

    Parameters
    ----------
    fp: str
        file path of the qiime2 feature table .qza artifact file
    sample_metadata_file : None or str, optional
        None (default) to just use sample names (no additional metadata).
        if not None, file path to the sample metadata (aka mapping file in QIIME).
    rep_seq_file: None or str, optional
        None (default) to use the feature ids in the feature table
        if not None, file path to the qiime2 representative sequences artifact file (defined by the qiime2 --o-representative-sequences parameter)
    taxonomy_file: None or str, optional
        if not None, add taxonomy for each feature using the qiime2 taxonomy artifact file (output of the qiime2 feature-classifier command)

    Keyword Arguments
    -----------------
    %(io.read.parameters)s
    '''
    newexp = read_amplicon(fp, sample_metadata_file=sample_metadata_file, data_file_type='qiime2', **kwargs)
    with tempfile.TemporaryDirectory() as tempdir:
        # if rep-seqs file is supplied, translate hashes to sequences
        if rep_seq_file is not None:
            logger.debug('loading rep_seqs file %s' % rep_seq_file)
            rs_name = _file_from_zip(tempdir, rep_seq_file, internal_data='data/dna-sequences.fasta')
            rseqs = []
            rids = []
            for chead, cseq in _iter_fasta(rs_name):
                rseqs.append(cseq.upper())
                rids.append(chead)
            rep_seqs = pd.Series(data=rseqs, index=rids, name='_feature_id')

            # test if all hashes are identical to the rep_seqs file supplied
            if not newexp.feature_metadata.index.equals(rep_seqs.index):
                logger.info('Rep seqs hashes and table hashes are not equal. Using table hashes.')
            # switch the columns so now _feature_id (and the index) is the sequence and not the hash. The hash is copied to '_hash'
            newexp.feature_metadata.rename(columns={'_feature_id': '_hash'}, inplace=True)
            newexp.feature_metadata = newexp.feature_metadata.join(other=rep_seqs, on='_hash', how='left')
            newexp.feature_metadata.set_index('_feature_id', inplace=True, drop=False)

        # if taxonomy file is supplied, load it into the feature metadata
        if taxonomy_file is not None:
            logger.debug('loading taxonomy file %s' % taxonomy_file)
            tax_name = _file_from_zip(tempdir, taxonomy_file, internal_data='data/taxonomy.tsv')
            taxonomy_df = pd.read_table(tax_name)
            taxonomy_df.set_index('Feature ID', inplace=True)
            newexp.feature_metadata = newexp.feature_metadata.join(other=taxonomy_df, how='left')
            if len(newexp.feature_metadata.index.intersection(taxonomy_df.index)) == 0:
                logger.info('No matching sequences in taxonomy file.')
                if '_hash' in newexp.feature_metadata.columns:
                    logger.info('Trying to use hashes for taxonomy')
                    newexp.feature_metadata = newexp.feature_metadata.drop(taxonomy_df.columns, axis=1)
                    newexp.feature_metadata = newexp.feature_metadata.join(other=taxonomy_df, on='_hash', how='left')
    return newexp


def _read_biom(fp, sample_in_row=False):
    '''Read in a biom table file.

    Parameters
    ----------
    fp : str or file object
        file path to the biom table
    sample_in_row: bool, optional
        False (defailt) - samples correspond to columns
        if True, samples in the biom table are in rows

    Returns
    -------
    sid : list of str
        the sample ids
    fid : list of str
        the feature ids
    data : numpy array (2d) of float
        the table
    feature_md : pandas.DataFrame
        the feature metadata (if availble in table)
    '''
    logger.debug('loading biom table %s' % fp)
    if hasattr(fp, 'read'):
        table = biom.parse_table(fp)
    else:
        table = biom.load_table(fp)
    if sample_in_row:
        table = table.transpose()
    sid = table.ids(axis='sample')
    fid = table.ids(axis='observation')
    logger.info('loaded %d samples, %d features' % (len(sid), len(fid)))
    data = table.matrix_data
    feature_md = _get_md_from_biom(table)

    # Transpose the data. The biom table has samples in columns while sklearn and other packages require samples in rows
    data = data.transpose()

    return sid, fid, data, feature_md


def _get_md_from_biom(table):
    '''Get the metadata of last column in the biom table.

    Return
    ------
    pandas.DataFrame or None
    '''
    ids = table.ids(axis='observation')
    metadata = table.metadata(axis='observation')
    if metadata is None:
        logger.debug('No metadata associated with features in biom table')
        md_df = None
    else:
        md_df = pd.DataFrame([dict(tmd) for tmd in metadata], index=ids)
    return md_df


def _read_csv(fp, sample_in_row=False, sep=','):
    '''Read a csv file

    Parameters
    ----------
    fp : str
        file path to the biom table
    sample_in_row : bool, optional
        True if csv datafile has samples as rows, features as columns
        False (default) if columns are samples (rows are features)
    sep : str, optional
        The separator between entries in the table

    Returns
    -------
    sid : list of str
        the sample ids
    fid : list of str
        the feature ids
    data : numpy array (2d) of float
        the table (samples in columns, features in rows)
    feature_md : pandas.DataFrame
        the feature metadata (if availble in table)
    '''
    logger.debug('loading csv table %s' % fp)
    # use the python engine as the default (c) engine throws an error
    # a known bug in pandas (see #11166)
    table = pd.read_csv(fp, header=0, engine='python', sep=sep)
    # if the csv file has an additional sep at the end of each line, it cause
    # pandas to create an empty column at the end. This can cause bugs with the
    # normalization. so we remove it.
    table.dropna(axis='columns', how='all', inplace=True)
    table.set_index(table.columns[0], drop=True, inplace=True)

    if sample_in_row:
        table = table.transpose()
        logger.debug('transposed table')
    sid = table.columns
    fid = table.index
    data = table.values.astype(float).transpose()
    logger.info('loaded %d samples, %d features' % data.shape)
    return sid, fid, data


def _read_metadata(ids, f, kwargs):
    '''read metadata table

    Parameters
    ----------
    ids : list like of str
        ids from data table
    f : str
        file path of metadata
    kwargs : dict
        keyword argument passed to :func:`pandas.read_csv`

    Returns
    -------
    pandas.DataFrame of metadata
    '''
    # load the sample/feature metadata file
    if f is None:
        metadata = pd.DataFrame(index=ids)
    else:
        if kwargs is None:
            kwargs = {}
        # the following code force the read the index_col as string
        # by reading it as a normal column and specify its dtype and then
        # setting it as index. There seems no better solution
        if 'index_col' in kwargs:
            index_col = kwargs.pop('index_col')
        else:
            index_col = 0
        if 'dtype' in kwargs:
            kwargs['dtype'][index_col] = str
        else:
            kwargs['dtype'] = {index_col: str}

        try:
            metadata = pd.read_csv(f, sep='\t', **kwargs)
        except Exception as err:
            logger.error('Error reading metadata file %r\nError: %s' % (f, err))
            raise err
        metadata.set_index(metadata.columns[index_col], inplace=True)
        mid, ids2 = set(metadata.index), set(ids)
        diff = mid - ids2
        if diff:
            logger.warning('These have metadata but do not have data - dropped (%d): %r' % (len(diff), diff))
        diff = ids2 - mid
        if diff:
            logger.warning('These have data but do not have metadata: %r' % diff)
        # reorder the id in metadata to align with biom
        # metadata = metadata.loc[ids, ]

        # check if we have duplicate index ids (which will raise)
        dup = metadata.index.duplicated(keep=False)
        if dup.sum() > 0:
            column_str = '%d' % index_col
            if metadata.index.name is not None:
                if len(metadata.index.name) > 0:
                    column_str = metadata.index.name
            errmsg = '%d duplicate id values encountered in index column %s of mapping file %r:' % (dup.sum(), column_str, f)
            errmsg += '\n%s' % set(metadata.index[dup].values)
            raise ValueError(errmsg)
        metadata = metadata.reindex(ids)
    return metadata


@ds.get_sectionsf('io.read')
def read(data_file, sample_metadata_file=None, feature_metadata_file=None,
         description='', sparse=True,
         data_file_type='biom', data_file_sep=',', sample_in_row=False,
         sample_id_proc=None, feature_id_proc=None,
         sample_metadata_kwargs=None, feature_metadata_kwargs=None,
         cls=Experiment,
         *, normalize):
    '''Read in an experiment.

    .. note:: The order in the sample and feature metadata tables are changed
       to align with data table.

    Parameters
    ----------
    data_file : str or file-like object.
        file path to the data table containing abundance values.
    sample_metadata_file : str, default=None
        File path to sample metadata (aka mapping file in
        QIIME). Default to just use sample names (no additional
        metadata).
    feature_metadata_file : str, default=None
        File path to feature metadata.
    description : str
        description of the experiment
    sparse : bool
        read the biom table into sparse or dense array
    data_file_type : str, default='biom'
        the data_file format. options:

        * 'biom': a biom table (biom-format.org)
        * 'csv': a comma-, tab-, or other delimiter- (as indicated by the
          `data_file_sep` parameter) separated table. By default,
          samples are in columns and features are in rows (see `sample_in_row` parameter).
        * 'qiime2': a qiime2 biom table artifact
    data_file_sep : str, default='\t'
        column delimitor for the data table if it is a 'csv' file.
    sample_in_row : bool, optional
        False if data table columns are sample, True if rows are samples
    sample_id_proc, feature_id_proc: callable, default=None
        If not `None`, modify each sample/feature id in the data table using
        the callable function. The callable accepts a list of str and
        returns a list of str (sample or feature ids after processing).
        Useful in metabolomics experiments, where the sample IDs in the
        data table contain additional information compared to the
        mapping file (using a '_' separator), and this needs to be
        removed in order to sync the sample IDs between data table and
        metadata file.
    sample_metadata_kwargs, feature_metadata_kwargs : dict, default=None
        keyword arguments passing to :func:`pandas.read_csv` when
        reading sample metadata or feature metadata. For example, you
        can set ``sample_metadata_kwargs={'dtype': {'pH': int},
        'encoding': 'latin-8'}`` to read the pH column in the sample
        metadata as int and parse the file as latin-8 instead of
        utf-8. By default, it assumes the first column in the metadata
        files is sample/feature IDs and is read in as row index. To
        avoid this, please provide {'index_col': False}.
    cls : ``class``, optional
        what class object to read the data into (:class:`.Experiment` by default)
    normalize : int or None
        normalize each sample to the specified read count. `None` to not normalize

    Returns
    -------
    Experiment
        the new object created

    '''
    # store the function parameters for call history
    fparams = locals()

    logger.debug('Reading experiment (%s, %s, %s)' % (
        data_file, sample_metadata_file, feature_metadata_file))

    # load the data table
    fmd = None
    if data_file_type == 'biom':
        sid, fid, data, fmd = _read_biom(data_file, sample_in_row=sample_in_row)
    elif data_file_type == 'csv':
        sid, fid, data = _read_csv(data_file, sample_in_row=sample_in_row, sep=data_file_sep)
    elif data_file_type == 'qiime2':
        sid, fid, data, fmd = _read_qiime2_zip(data_file)
    elif data_file_type == 'tsv':
        sid, fid, data = _read_csv(data_file, sample_in_row=sample_in_row, sep='\t')
    else:
        raise ValueError('unkown data_file_type %s' % data_file_type)

    sid = [str(x) for x in sid]

    # if we need to process the table sample/feature IDs
    if sample_id_proc is not None:
        sid = sample_id_proc(sid)
    if feature_id_proc is not None:
        fid = feature_id_proc(fid)

    sample_metadata = _read_metadata(sid, sample_metadata_file, sample_metadata_kwargs)
    feature_metadata = _read_metadata(fid, feature_metadata_file, feature_metadata_kwargs)

    # store the sample and feature ids also as a column (for sorting, etc.)
    sample_metadata['_sample_id'] = sample_metadata.index.values
    feature_metadata['_feature_id'] = feature_metadata.index.values

    # store the abundance per sample/feature before any procesing
    sample_metadata['_calour_original_abundance'] = data.sum(axis=1)
    # self.feature_metadata['_calour_original_abundance'] = self.data.sum(axis=0)

    if fmd is not None:
        # rename columns in biom table if exist in feature metadata file
        renames = {}
        for ccol in fmd.columns:
            if ccol in feature_metadata.columns:
                renames[ccol] = ccol + '_biom'
            if renames:
                fmd.rename(columns=renames, inplace=True)
        # combine it with the feature metadata
        feature_metadata = pd.concat([feature_metadata, fmd], axis=1)

    # init the experiment details
    info = {'data_file': data_file,
            'data_md5': get_data_md5(data),
            'sample_metadata_file': sample_metadata_file,
            'sample_metadata_md5': get_file_md5(sample_metadata_file),
            'feature_metadata_file': feature_metadata_file,
            'feature_metadata_md5': get_file_md5(feature_metadata_file)}

    exp = cls(data, sample_metadata, feature_metadata,
              info=info, description=description, sparse=sparse)

    if normalize is not None:
        exp.normalize(total=normalize, inplace=True)

    # initialize the call history
    param = ['{0!s}={1!r}'.format(k, v) for k, v in fparams.items()]
    exp._call_history = ['{0}({1})'.format('read_amplicon', ','.join(param))]

    return exp


@ds.with_indent(4)
def read_amplicon(data_file, sample_metadata_file=None,
                  *, min_reads, normalize, **kwargs):
    '''Read in an amplicon experiment.

    This wraps :func:`read`.

    Parameters
    ----------
    min_reads : int or None
        remove all samples with less than `min_reads`. `None` to keep all samples

    Keyword Arguments
    -----------------
    %(io.read.parameters)s

    Returns
    -------
    AmpliconExperiment
        after removing low read sampls and normalization

    See Also
    --------
    read

    '''
    # store the function parameters for call history
    fparams = locals()

    # don't do normalize before the possible filtering
    exp = read(data_file, sample_metadata_file, cls=AmpliconExperiment,
               normalize=None, sample_in_row=False, **kwargs)

    if 'taxonomy' in exp.feature_metadata.columns:
        exp.feature_metadata['taxonomy'] = _get_taxonomy_string(exp, remove_underscore=False)

    if min_reads is not None:
        exp.filter_by_data('abundance', axis=0, cutoff=min_reads, mean_or_sum='sum', inplace=True)
    if normalize is not None:
        exp.normalize(total=normalize, axis=0, inplace=True)

    # initialize the call history
    param = ['{0!s}={1!r}'.format(k, v) for k, v in fparams.items()]
    exp._call_history = ['{0}({1})'.format('read_amplicon', ','.join(param))]

    return exp


@ds.with_indent(4)
def read_ms(data_file, sample_metadata_file=None, feature_metadata_file=None, gnps_file=None,
            data_file_type='mzmine2', sample_in_row=None, direct_ids=None, get_mz_rt_from_feature_id=None,
            use_gnps_id_from_AllFiles=True, cut_sample_id_sep=None,
            mz_rt_sep=None, mz_thresh=0.02, rt_thresh=15,
            description=None, sparse=False, *, normalize, **kwargs):
    '''Read a mass-spec experiment.

    Calour supports various ms table formats, with several preset formats (specified by the data_file_type='XXX' parameter),
    as well as able to read user specified formats.

    With the installation of the gnps-calour database interface, Calour can integrate MS2 information from GNPS into the analysis:

    If the data table and the gnps file share the same IDs (preferred), GNPS annotations use the uniqueID of the features. Otherwise, calour
    matches the features to the gnps file using an MZ and RT threshold window (specified by the mz_thresh=XXX, rt_thresh=XXX parameters).

    Supported formats for ms analysis (as specified by the data_file_type='XXX' parameter) include:

    * 'mzmine2': using the csv output file of mzmine2. MZ and RT are obtained via the 2nd and 3rd column in the file.

    * 'biom': using a biom table for the metabolite table. featureIDs in the table (first column) can be either MZ_RT (concatenated with a separator), or a unique ID matching the gnps_file ids.

    * 'openms': using a csv data table with MZ_RT or unqie ID as featureID (first column). samples can be columns (default) or rows (using the sample_in_row=True parameter)

    * 'gnps-ms2': a tsv file exported from gnps, with gnps ids as featureIDs.

    Parameters
    ----------
    data_file : str
        The name of the data table (mzmine2 output/bucket table/biom table) containing the per-metabolite abundances.
    sample_metadata_file : str or None (optional)
        None (default) to not load metadata per sample
        str to specify name of sample mapping file (tsv).

        Note: sample names in the bucket table and sample_metadata file must match. In case bucket table sample names contains additional
        information, you can split them at the separator character (usually '_'), keeping only the first part, using the cut_sample_id_sep='_' parameter
        (see below)
    gnps_file : str or None (optional)
        name of the gnps clusterinfosummarygroup_attributes_withIDs_arbitraryattributes/XXX.tsv file, for use with the 'gnps' database.
        This enables identification of the metabolites with known MS2 (for the interactive heatmap and sorting/filtering etc), as well as linking
        to the gnps page for each metabolite (from the interactive heatmap - by double clicking on the metabolite database information).
        Note: requires gnps-calour database interface module (see Calour installation instructions for details).
    feature_metadata_file : str or None (optional)
        Name of table containing additional metadata about each feature
        None (default) to not load
    data_file_type: str, optional
        the data file format. options include:

        'mzmine2': load the mzmine2 output csv file.
            MZ and RT are obtained from this file.
            GNPS linking is direct via the unique id column.
            table is csv, columns are samples.
        'biom': load a biom table for the features
            MZ and RT are obtained via the featureID (first column), which is assumed to be MZ_RT.
            GNPS linking is indirect via the mz and rt threshold windows.
            table is a tsv/json/hdf5 biom table, columns are samples.
        'openms': load an openms output table
            MZ and RT are obtained via the featureID (first column), which is assumed to be MZ_RT.
            GNPS linking is indirect via the mz and rt threshold windows.
            table is a csv table, columns are samples.
        'gnps-ms2': load a gnps exported biom table
            MZ and RT are obtained via the gnps_file if available, otherwise are NA
            GNPS linking is direct via the first column (featureID).
            table is a tsv/json/hdf5 biom table, columns are samples.
    sample_in_row: bool or None, optional
        False indicates rows in the data table file are features, True indicates rows are samples.
        None to use default value according to data_file_type
    direct_ids: bool or None, optional
        True indicates the feature ids in the data table file are the same ids used in the gnps_file.
        False indicates feature ids are not the same as in the gnps_file (such as when the ids are the MZ_RT)
        None to use default value according to data_file_type
    get_mz_rt_from_feature_id: bool or None, optional
        True indicates the data table file feature ids contain the MZ/RT of the feature.
        False to not obtain MZ/RT from the feature id
        None to use default value according to data_file_type
    use_gnps_id_from_AllFiles: bool, optional
        True (default) to link the data table file gnps ids to the AllFiles column in the gnps_file.
        False to link the data table file gnps ids to the 'cluster index' column in the gnps_file.
    cut_sample_id_sep: str or None, optional
        str (typically '_') to split the sampleID in the data table file, keeping only the first part.
        Useful when the sampleIDs in the data table contain additional information compared to the
        mapping file (using a '_' separator), and this needs to be removed in order to sync the sampleIDs between table and mapping file.
        None (default) to not change the data table file sampleID
    mz_rt_sep: str or None, optional
        The separator character between the MZ and RT parts of the featureID (if it contains them) (usually '_').
        If not supplied, autodetect the separator.
        Note this is used only if get_mz_rt_from_feature_id=True
    mz_thresh: float, optional
        The tolerance for M/Z to match features to the gnps_file. Used only if parameter direct_ids=False.
    rt_thresh: float, optional
        The tolerance for retention time to match features to the gnps_file. Used only if parameter direct_ids=False.
    description : str or None (optional)
        Name of the experiment (for display purposes).
        None (default) to assign file name
    sparse : bool (optional)
        False (default) to store data as dense matrix (faster but more memory)
        True to store as sparse (CSR)
    normalize : int or None
        normalize each sample to the specified reads. `None` to not normalize

    Keyword Arguments
    -----------------
    %(io.read.parameters)s

    Returns
    -------
    MS1Experiment

    See Also
    --------
    read
    '''
    # store the function parameters for call history
    fparams = locals()

    default_params = {'mzmine2': {'sample_in_row': False, 'direct_ids': True, 'get_mz_rt_from_feature_id': False, 'ctype': 'csv'},
                      'biom': {'sample_in_row': False, 'direct_ids': False, 'get_mz_rt_from_feature_id': True, 'ctype': 'biom'},
                      'openms': {'sample_in_row': False, 'direct_ids': False, 'get_mz_rt_from_feature_id': True, 'ctype': 'csv'},
                      'gnps-ms2': {'sample_in_row': False, 'direct_ids': True, 'get_mz_rt_from_feature_id': False, 'ctype': 'biom'}}

    if data_file_type not in default_params:
        raise ValueError('data_file_type %s not recognized. valid options are: %s' % (data_file_type, default_params.keys()))

    # set the default params according to file type, if not specified by user
    if sample_in_row is None:
        sample_in_row = default_params[data_file_type]['sample_in_row']
    if direct_ids is None:
        direct_ids = default_params[data_file_type]['direct_ids']
    if get_mz_rt_from_feature_id is None:
        get_mz_rt_from_feature_id = default_params[data_file_type]['get_mz_rt_from_feature_id']

    logger.debug('Reading MS data (data table %s, map file %s, data_file_type %s)' % (data_file, sample_metadata_file, data_file_type))
    exp = read(data_file, sample_metadata_file, feature_metadata_file,
               data_file_type=default_params[data_file_type]['ctype'], sparse=sparse,
               normalize=None, cls=MS1Experiment,
               sample_id_proc=lambda x: _split_sample_ids(x, split_char=cut_sample_id_sep),
               sample_in_row=sample_in_row, **kwargs)
    # get the MZ/RT data
    if data_file_type == 'mzmine2':
        if 'row m/z' not in exp.sample_metadata.index:
            raise ValueError('Table file %s does not contain "row m/z" column. Is it an mzmine2 data table?' % data_file)
        mzpos = exp.sample_metadata.index.get_loc('row m/z')
        if 'row retention time' not in exp.sample_metadata.index:
            raise ValueError('Table file %s does not contain "row retention time" column. Is it an mzmine2 data table?' % data_file)
        rtpos = exp.sample_metadata.index.get_loc('row retention time')
        # get the MZ and RT
        exp.feature_metadata['MZ'] = exp.data[mzpos, :]
        exp.feature_metadata['RT'] = exp.data[rtpos, :]
        # drop the two columns which are not samples
        sample_pos = np.arange(len(exp.sample_metadata))
        sample_pos = list(set(sample_pos).difference([mzpos, rtpos]))
        exp = exp.reorder(sample_pos)
    if get_mz_rt_from_feature_id:
        logger.debug('getting MZ and RT from featureIDs')
        if direct_ids:
            raise ValueError('Cannot get mz/rt from feature ids if direct_ids=True.')
        # if needed, autodetect the mz/rt separator
        if mz_rt_sep is None:
            logger.debug('autodetecting mz/rt separator')
            tmp = exp.feature_metadata['_feature_id'].iloc[0].split('_')
            if len(tmp) == 2:
                logger.debug('Autodetcted "_" as mz/rt separator')
                mz_rt_sep = '_'
            else:
                tmp = exp.feature_metadata['_feature_id'].iloc[0].split()
                if len(tmp) == 2:
                    logger.debug('Autodetcted " " as mz/rt separator')
                    mz_rt_sep = None
                else:
                    raise ValueError('No separator detected for mz/rt separation in feature ids. please specify separator in mz_rt_sep parameter')
        # get the MZ/RT
        try:
            exp.feature_metadata[['MZ', 'RT']] = exp.feature_metadata['_feature_id'].str.split(mz_rt_sep, expand=True)
        except ValueError:
            raise ValueError('Failed to obtain MZ/RT from feature ids. Maybe use get_mz_rt_from_feature_id=False?')

        # mz and rt are numbers
        exp.feature_metadata['MZ'] = exp.feature_metadata['MZ'].astype(float)
        exp.feature_metadata['RT'] = exp.feature_metadata['RT'].astype(float)

    if normalize is not None:
        # record the original total read count into sample metadata
        exp.normalize(total=normalize, inplace=True)

    # Create the combined field for easy sorting/plotting
    if 'MZ' in exp.feature_metadata and 'RT' in exp.feature_metadata:
        exp.feature_metadata['mz_rt'] = ['%08.4f_%05.2f' % (x[1]['MZ'], x[1]['RT']) for x in exp.feature_metadata.iterrows()]

    if gnps_file:
        # load the gnps table
        gnps_data = pd.read_csv(gnps_file, sep='\t')
        exp.info['_calour_metabolomics_gnps_table'] = gnps_data
        # use the gnpscalour database interface to get metabolite info from the gnps file
        gnps_db = _get_database_class('gnps', exp=exp)
        # link each feature to the gnps ids based on MZ/RT or direct_id
        gnps_db._prepare_gnps_ids(direct_ids=direct_ids, mz_thresh=mz_thresh, use_gnps_id_from_AllFiles=use_gnps_id_from_AllFiles)
        # add gnps names and cluster to the features as feature_metadata fields (gnps_name and gnps_cluster)
        gnps_db._prepare_gnps_names()

    # initialize the call history
    param = ['{0!s}={1!r}'.format(k, v) for k, v in fparams.items()]
    exp._call_history = ['{0}({1})'.format('read_amplicon', ','.join(param))]
    return exp


def _split_sample_ids(sid, split_char=None, ignore_split=('row m/z', 'row retention time')):
    '''Split the sample id in data table using the split_char returning the first split str.

    Used in the read_ms() function, as a callable for the read() function

    Parameters
    ----------
    sid : list of str
        the list of sample ids to process
    split_char: str or None, optional
        None to not split the sampleids
        str to split sample id using this string

    Returns
    -------
    list of str: the split sample ids
    '''
    if split_char is None:
        return sid
    logger.info('splitting table sample ids using separator %s. use "data_table_params={\'cut_sample_id_sep\'=None}" to disable cutting.' % split_char)
    return [x.split(split_char)[0] if x not in ignore_split else x for x in sid]


def save(exp: Experiment, prefix, fmt='hdf5'):
    '''Save the experiment data to disk.

    This saves data table, sample metadata, and feature metadata to 3
    different files with the same file prefix.

    Parameters
    ----------
    prefix : str
        file path (suffixes auto added for the 3 files) to save to.
    fmt : str
        format for the data table. could be 'hdf5', 'txt', or 'json'.

    '''
    exp.save_biom('%s.biom' % prefix, fmt=fmt)
    exp.sample_metadata.to_csv('%s_sample.txt' % prefix, sep='\t')
    exp.feature_metadata.to_csv('%s_feature.txt' % prefix, sep='\t')


def save_biom(exp: Experiment, f, fmt='hdf5', add_metadata='taxonomy'):
    '''Save experiment to biom format

    Parameters
    ----------
    f : str
        the file to save to
    fmt : str, optional
        the output biom table format. options are:
        'hdf5' (default) save to hdf5 biom table.
        'json' same to json biom table.
        'txt' save to text (tsv) biom table.
    add_metadata : str or None, optional
        add metadata column from `Experiment.feature_metadata` to biom table.
        Don't add if it is `None`.

    '''
    logger.debug('save biom table to file %s format %s' % (f, fmt))
    if fmt == 'hdf5':
        tab = _create_biom_table_from_exp(exp, add_metadata, to_list=True)
        with biom.util.biom_open(f, 'w') as f:
            tab.to_hdf5(f, "calour")
    elif fmt == 'json':
        tab = _create_biom_table_from_exp(exp, add_metadata)
        with open(f, 'w') as f:
            tab.to_json("calour", f)
    elif fmt == 'txt':
        tab = _create_biom_table_from_exp(exp, add_metadata)
        if add_metadata:
            logger.warning('.txt format does not support taxonomy information in save. Saving without taxonomy.')
        s = tab.to_tsv()
        with open(f, 'w') as f:
            f.write(s)
    else:
        raise ValueError('Unknown file format %s for save' % fmt)
    logger.debug('biom table saved to file %s' % f)


def save_fasta(exp: Experiment, f, seqs=None, header='seq'):
    '''Save a list of sequences to fasta.

    Use taxonomy information if available, otherwise just use sequence as header.

    Parameters
    ----------
    f : str
        the filename to save to
    seqs : list of str sequences ('ACGT') or None, optional
        None (default) to save all sequences in exp, or list of sequences to only save these sequences.
        Note: sequences not in exp will not be saved
    header: str, optional
        The format for the per-sequence header in the output fasta file. options are:
        'seq' (default): use the actual sequence as header
        'num': use the 'Seq_i' as header (where i is the enumerated index).
    '''
    logger.debug('Save seq to fasta file %s' % f)
    if seqs is None:
        logger.debug('no sequences supplied - saving all sequences')
        seqs = exp.feature_metadata.index.values
    num_skipped = 0
    with open(f, 'w') as fasta_file:
        for idx, cseq in enumerate(seqs):
            if cseq not in exp.feature_metadata.index:
                num_skipped += 1
                continue
            if header == 'seq':
                cheader = cseq
            elif header == 'num':
                cheader = 'Seq_%d' % idx
            else:
                raise ValueError('header %s not supported' % header)
            fasta_file.write('>%s\n%s\n' % (cheader, cseq))
    logger.debug('wrote fasta file with %d sequences. %d sequences skipped' % (len(seqs) - num_skipped, num_skipped))


def _create_biom_table_from_exp(exp, add_metadata='taxonomy', to_list=False):
    '''Create a biom table from an experiment

    Parameters
    ----------
    exp : Experiment
    add_metadata : str or None, optional
        add metadata column from `Experiment.feature_metadata` to biom table.
        Don't add if it is `None`.
    to_list: bool, optional
        True to convert the metadata field to list (for hdf5)

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
        if add_metadata in exp.feature_metadata.columns:
            # md has to be a dict of dict, so it needs to be converted from
            # a DataFrame instead of Series
            md = exp.feature_metadata.loc[:, [add_metadata]].to_dict('index')
            # we need to make it into a list of taxonomy levels otherwise biom save fails for hdf5
            if to_list:
                for k, v in md.items():
                    # if isinstance(v[add_metadata], str):
                    v[add_metadata] = v[add_metadata].split(';')
            table.add_metadata(md, axis='observation')
        else:
            logger.info('Metadata field %s not found. Saving biom table without metadata' % add_metadata)
    return table


def _iter_fasta(fp):
    '''Read fasta file into iterator.

    Fasta file must contain header line (starting with ">") and one or more sequence lines.

    Parameters
    ----------
    fp: str
        name of the fasta file

    Yields
    ------
    header: str
        the header line (without ">")
    sequence: str
        the sequence ('ACGT'). Both header and sequence are whitespace stripped.
    '''
    # skip non-header lines at beginning of file
    with open(fp, 'r') as fl:
        for cline in fl:
            if cline[0] == ">":
                title = cline[1:].rstrip()
                break
            logger.warning('Fasta file %s has no headers' % fp)
            return

        lines = []
        for cline in fl:
            if cline[0] == ">":
                yield title, ''.join(lines)
                lines = []
                title = cline[1:].strip()
                continue
            lines.append(cline.strip())
        yield title, "".join(lines)
