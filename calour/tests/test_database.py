# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from os.path import join
from tempfile import mkdtemp
import shutil
import logging

from calour._testing import Tests
from calour.tests.mock_database import MockDatabase
from calour.heatmap.heatmap import _create_plot_gui
import calour.util
from calour.database import _get_database_class
import calour as ca


class ExperimentTests(Tests):
    def setUp(self):
        super().setUp()
        self.mock_db = MockDatabase()
        self.test1 = ca.read_amplicon(self.test1_biom, self.test1_samp,
                                      min_reads=1000, normalize=10000)
        self.s1 = 'TACGTATGTCACAAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCCGTGGATTAAGCGTGTTGTGAAATGTAGACGCTCAACGTCTGAATCGCAGCGCGAACTGGTTCACTTGAGTATGCACAACGTAGGCGGAATTCGTCG'

    def test_mock_db(self):
        mdb = self.mock_db
        self.assertTrue(mdb.annotatable)
        self.assertTrue(mdb.can_do_enrichment)
        self.assertEqual(mdb.database_name, 'mock_db')

    def test_gui_interface(self):
        mdb = self.mock_db
        res = mdb.get_seq_annotation_strings(self.s1)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[1][1], 'nice')
        self.assertFalse('_db_interface' in res[0][0])
        gui = _create_plot_gui(self.test1, gui='qt5', databases=[])
        gui.databases.append(mdb)
        res = gui.get_database_annotations(self.s1)
        self.assertEqual(len(res), 2)
        self.assertTrue('_db_interface' in res[0][0])
        self.assertEqual(res[1][1], 'nice')

    def test_get_database_class(self):
        d = mkdtemp()
        f = join(d, 'config.txt')
        calour.util.set_config_value('class_name', 'MockDatabase', section='testdb', config_file_name=f)
        calour.util.set_config_value('module_name', 'calour.tests.mock_database', section='testdb', config_file_name=f)
        db = _get_database_class('testdb', config_file_name=f)
        self.assertEqual(db.database_name, 'mock_db')

        # test None results if database does not exist in config file
        res = _get_database_class('mock')
        self.assertEqual(res, None)

        shutil.rmtree(d)

    def test_get_database_class_version(self):
        d = mkdtemp()
        f = join(d, 'config.txt')
        calour.util.set_config_value('class_name', 'MockDatabase', section='testdb', config_file_name=f)
        calour.util.set_config_value('module_name', 'calour.tests.mock_database', section='testdb', config_file_name=f)
        calour.util.set_config_value('min_version', '9999.9999', section='testdb', config_file_name=f)
        # re-enable logging because it is disabled in setUp
        logging.disable(logging.NOTSET)
        with self.assertLogs(level='WARNING') as cm:
            db = _get_database_class('testdb', config_file_name=f)
            self.assertEqual(db.database_name, 'mock_db')
            self.assertRegex(cm.output[0], 'Please update')
        shutil.rmtree(d)


if __name__ == "__main__":
    main()
