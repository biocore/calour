# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

from calour._testing import Tests
import calour as ca


class Ms1ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.ms1 = ca.read_open_ms(self.openms_csv, None, gnps_file=self.ms1_gnps, normalize=None)

    def test_prepare_gnps(self):
        self.assertIn('gnps', self.ms1.feature_metadata)


if __name__ == "__main__":
    main()
