# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from os.path import join

import pandas as pd
import numpy as np
import scipy
import biom


logger = getLogger(__name__)


class Study:
    '''This class contains the data for a study or a meta study.

    The data set includes a data table (otu table, gene table,
    metabolomic table, or all those tables combined), a sample
    metadata table, and a feature metadata.

    Parameters
    ----------
    data : ``numpy.array`` or ``scipy.sparse``

    sample_metadata : ``pandas.DataFrame``

    feature_metadata : ``pandas.DataFrame``
    '''
    def __init__(self, data, sample_metadata, feature_metadata=None,
                 description='', sparse=True):
        self.data = data
        self.sample_metadata = sample_metadata
        self.feature_metadata = feature_metadata
        self.description = description

    @staticmethod
    def _read_biom(fp, transpose=True, sparse=True):
        '''Read in a biom table file.

        Parameters
        ----------
        fp : str
            file path to the biom table
        transpose : bool
            Transpose the table or not. The OTU table has samples in
            column while sklearn and other packages require samples in
            row. So you should transpose the data table.
        '''
        table = biom.load_table(fp)
        sid = table.ids(axis='sample')
        oid = table.ids(axis='observation')
        if sparse:
            data = scipy.sparse.csr_matrix(table.matrix_data)
        else:
            data = table.matrix_data.toarray()

        feature_md = _get_md_from_biom(table)

        if transpose:
            data = data.transpose()

        return sid, oid, data, feature_md

    @staticmethod
    def _get_md_from_biom(table):
        '''Get the metadata of last column in the biom table.

        Return
        ------
        pandas.DataFrame
        '''

        return md

    @staticmethod
    def _read_table(f):
        '''Read tab-delimited table file.

        It is used to read sample metadata (mapping) file and feature
        metadata file

        '''
        table = pd.read_table(f, sep='\t', index_col=0)
        # make sure the sample ID is string-type
        table.index = table.index.astype(np.str)
        return table

    def __repr__(self):
        '''Representation of this object.'''


    @classmethod
    def read(cls, data, sample_metadata=None, feature_metadata=None,
             description='', sparse=True):
        '''Read the files for the study.

        Parameters
        ----------
        data : str
            file path to the biom table.
        sample_metadata : str
            file path to the sample metadata (aka mapping file in QIIME)
        feature_metadata : str
            file path to the feature metadata.
        description : str
            description of the study
        sparse : bool
            read the biom table into sparse or dense array
        '''
        logger.info('Reading the study files...')
        sid, oid, data, md = cls._read_biom(data)
        if sample_metadata is not None:
            # reorder the sample id to align with biom
            sample_metadata = cls._read_table(sample_metadata).loc[sid, ]
        if feature_metadata is not None:
            # reorder the feature id to align with that from biom table
            fm = cls._read_table(feature_metadata).loc[oid, ]
            # combine it with the metadata from biom
            feature_metadata = pd.concat([fm, md], axis=1)
        else:
            feature_metadata = md
        return cls(data, sample_metadata, feature_metadata,
                   description=description, sparse=sparse)

    def save(self, f):
        '''Save the study data to disk.
        Parameters
        ----------
        f : str
            file path to save to.
        '''
