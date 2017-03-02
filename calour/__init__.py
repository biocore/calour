# ----------------------------------------------------------------------------
# Copyright (c) 2016--, calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from logging.config import fileConfig

from pkg_resources import resource_filename

from .experiment import Experiment, add_functions
from .amplicon_experiment import AmpliconExperiment
from .io import read, read_amplicon, read_open_ms
from .util import set_log_level


__credits__ = "https://github.com/biocore/calour/graphs/contributors"
__version__ = "0.1.0.dev0"

__all__ = ['read', 'read_amplicon', 'read_open_ms',
           'Experiment', 'AmpliconExperiment',
           'set_log_level']

add_functions(Experiment)

log = resource_filename(__package__, 'log.cfg')

# setting False allows other logger to print log.
fileConfig(log, disable_existing_loggers=False)
