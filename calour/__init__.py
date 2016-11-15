# ----------------------------------------------------------------------------
# Copyright (c) 2016--, calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

__credits__ = "https://github.com/biocore/calour/graphs/contributors"
__version__ = "0.1.0.dev0"


from calour.experiment import Experiment, add_functions
from calour.io import read
from calour.util import _get_taxonomy_string
from calour.normalization import normalize

add_functions(Experiment)
