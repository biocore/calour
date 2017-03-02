# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

from calour._testing import Tests
from calour.tests.mock_database import MockDatabase
from calour.heatmap.heatmap import create_plot_gui
import calour as ca


class ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.mock_db = MockDatabase()
        self.test1 = ca.read_amplicon(self.test1_biom, self.test1_samp, normalize=True)
        self.s1 = 'TACGTATGTCACAAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCCGTGGATTAAGCGTGTTGTGAAATGTAGACGCTCAACGTCTGAATCGCAGCGCGAACTGGTTCACTTGAGTATGCACAACGTAGGCGGAATTCGTCG'

    def test_mock_db(self):
        mdb = self.mock_db
        self.assertTrue(mdb.annotatable)
        self.assertTrue(mdb.can_get_feature_terms)
        self.assertEqual(mdb.get_name(), 'mock_db')

    def test_gui_interface(self):
        mdb = self.mock_db
        res = mdb.get_seq_annotation_strings(self.s1)
        print(res)
        gui = create_plot_gui(self.test1, gui='qt5', databases=[])
        gui.databases.append(mdb)
        res = gui.get_database_annotations(self.s1)
        print(res)


if __name__ == "__main__":
    main()
