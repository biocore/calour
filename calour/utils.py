# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging import getLogger

import numpy as np


logger = getLogger(__name__)


def get_fields(exp):
	'''
	return the sample fields of an experiment
	'''
	return list(exp.sample_metadata.columns)


def get_field_vals(exp,field,unique=True):
	'''
	return the values in sample field (unique or all)
	'''
	vals = exp.sample_metadata[field]
	if unique:
		vals = list(set(vals))
	return vals
