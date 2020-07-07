# ----------------------------------------------------------------------------
# Copyright (c) 2016--, calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging.config import fileConfig

from pkg_resources import resource_filename

from .experiment import Experiment
from .amplicon_experiment import AmpliconExperiment
from .ms1_experiment import MS1Experiment
from .io import read, read_amplicon, read_ms, read_qiime2
from .util import set_log_level, register_functions


__credits__ = "https://github.com/biocore/calour/graphs/contributors"
__version__ = "2019.5.1"

__all__ = ['read', 'read_amplicon', 'read_ms', 'read_qiime2',
           'Experiment', 'AmpliconExperiment', 'MS1Experiment',
           'set_log_level']


# add member functions to the class
register_functions((Experiment, AmpliconExperiment, MS1Experiment))


# setting False allows other logger to print log.
fileConfig(resource_filename(__package__, 'log.cfg'), disable_existing_loggers=False)
