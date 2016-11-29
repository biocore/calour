# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger
from unittest import TestCase, main
from os.path import join, dirname, abspath

import numpy as np


logger = getLogger(__name__)


def get_fields(exp):
    '''
    return the sample fields of an experiment
    '''
    return list(exp.sample_metadata.columns)


def get_field_vals(exp, field, unique=True):
    '''
    return the values in sample field (unique or all)
    '''
    vals = exp.sample_metadata[field]
    if unique:
        vals = list(set(vals))
    return vals


def _get_taxonomy_string(exp, separator=';', remove_underscore=True, to_lower=False):
    '''Get a nice taxonomy string
    Convert the taxonomy list stored (from biom.read_table) to a single string per feature

    Parameters
    ----------
    exp : Experiment
        with the taxonomy entry in the feature_metadata
    separator : str (optional)
        the output separator to use between the taxonomic levels
    remove_underscore : bool (optional)
        True (default) to remove the 'g__' entries and missing values
        False to keep them
    to_lower : bool (optional)
        False (default) to keep case
        True to convert to lowercase

    Returns
    -------
    taxonomy : list of str
        list of taxonomy string per feature
    '''
    # test if we have taxonomy in the feature metadata
    logger.debug('getting taxonomy string')
    if 'taxonomy' not in exp.feature_metadata.columns:
        raise ValueError('No taxonomy field in experiment')

    if not remove_underscore:
        taxonomy = [separator.join(x) for x in exp.feature_metadata['taxonomy']]
    else:
        taxonomy = []
        for ctax in exp.feature_metadata['taxonomy']:
            taxstr = ''
            for clevel in ctax.split(';'):
                clevel = clevel.strip()
                if len(clevel) > 3:
                    if clevel[1:3] == '__':
                        clevel = clevel[3:]
                    taxstr += clevel + separator
            if len(taxstr) == 0:
                taxstr = 'na'
            taxonomy.append(taxstr)

    if to_lower:
        taxonomy = [x.lower() for x in taxonomy]
    return taxonomy
