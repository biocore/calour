# ----------------------------------------------------------------------------
# Copyright (c) 2016--, calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging.config import fileConfig

import importlib.resources as pkg_resources

from .experiment import Experiment
from .amplicon_experiment import AmpliconExperiment
from .correlation_experiment import CorrelationExperiment
from .ms1_experiment import MS1Experiment
from .mrna_experiment import mRNAExperiment
from .io import read, read_amplicon, read_ms, read_qiime2, read_correlation
from .util import set_log_level, register_functions


__credits__ = "https://github.com/biocore/calour/graphs/contributors"
__version__ = "2024.8.25"

__all__ = ['read', 'read_amplicon', 'read_ms', 'read_qiime2', 'read_correlation',
           'Experiment', 'AmpliconExperiment', 'MS1Experiment','mRNAExperiment',
           'CorrelationExperiment',
           'set_log_level']


# add member functions to the class
register_functions((Experiment, AmpliconExperiment, MS1Experiment, mRNAExperiment, CorrelationExperiment))


# setting False allows other logger to print log.
log_cfg = pkg_resources.files('calour').joinpath('log.cfg')
with log_cfg.open('r') as config_file:
    fileConfig(config_file, disable_existing_loggers=False)
