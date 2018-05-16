# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from tempfile import mkdtemp
from os.path import join
import shutil

from calour._testing import Tests
import calour as ca


class ExportHtmlTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)

    def test_export_html(self):
        exp = self.test1
        d = mkdtemp()
        f = join(d, 'calour_interactive_heatmap.html')
        exp.export_html(sample_field='group', output_file=f, feature_field='taxonomy', title='the heatmap')

        # to create the file for testing visually
        # (needs to be manually opened and inspected before accepting)
        # exp.export_html(sample_field='group', output_file=join(self.test_data_dir, 'export_html_result.html'), feature_field='taxonomy', title='the heatmap')

        # load the results
        with open(f) as output_fl:
            output_data = output_fl.readlines()
        with open(join(self.test_data_dir, 'export_html_result.html')) as expected_fl:
            expected_data = expected_fl.readlines()
        # we cannot test equality of the whole files since plot output is different
        # on different machines
        self.assertEqual(len(output_data), len(expected_data))
        self.assertEqual(output_data[:100], expected_data[:100])

        shutil.rmtree(d)


if __name__ == "__main__":
    main()
