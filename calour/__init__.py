# ----------------------------------------------------------------------------
# Copyright (c) 2016--, calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from calour.experiment import Experiment, add_functions
from calour.io import read, read_bacteria


__credits__ = "https://github.com/biocore/calour/graphs/contributors"
__version__ = "0.1.0.dev0"

# link these functions to calour module
__all__ = [read, read_bacteria]

# add the function as normal class methods to Experiment
add_functions(Experiment)
