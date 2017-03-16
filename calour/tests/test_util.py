# ----------------------------------------------------------------------------
# Copyright (c) 2016--,  Calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main
from copy import deepcopy
from os.path import basename, join
from tempfile import mkdtemp
import shutil

import calour as ca
from calour import util

from calour._testing import Tests


class IOTests(Tests):
    def setUp(self):
        super().setUp()
        self.test1 = ca.read(self.test1_biom, self.test1_samp, normalize=None)

    def test_get_taxonomy_string(self):
        orig_tax = list(self.test1.feature_metadata['taxonomy'].values)
        # test string taxonomy
        tax = util._get_taxonomy_string(self.test1)
        self.assertListEqual(tax, orig_tax)
        # test using a biom table with list taxonomy, not removing the X__ parts
        exp = deepcopy(self.test1)
        exp.feature_metadata['taxonomy'] = exp.feature_metadata['taxonomy'].str.split(';')
        tax = util._get_taxonomy_string(exp, remove_underscore=False)
        self.assertListEqual(tax, orig_tax)
        # and test with removing the X__ parts and lower case
        tax = util._get_taxonomy_string(exp, to_lower=True)
        self.assertEqual(tax[1], 'bacteria;tenericutes;mollicutes;mycoplasmatales;mycoplasmataceae;mycoplasma;')

    def test_get_file_md5(self):
        md5 = util.get_file_md5(self.test1_samp)
        self.assertEqual(md5, 'c109f0e9cb6cd24ae820ea9c4bf84931')

    def test_get_data_md5(self):
        exp = deepcopy(self.test1)
        # try on dense matrix
        exp.sparse = True
        md5 = util.get_data_md5(exp.data)
        self.assertEqual(md5, '561ba229f4a4c68979e560a10cc3fe42')

        # try on sparse matrix
        exp.sparse = False
        md5 = util.get_data_md5(exp.data)
        self.assertEqual(md5, '561ba229f4a4c68979e560a10cc3fe42')

    def test_get_config_file(self):
        fp = util.get_config_file()
        self.assertEqual(basename(fp), 'calour.config')

    def test_get_config_sections(self):
        sections = util.get_config_sections()
        self.assertIn('dbbact', sections)
        self.assertIn('sponge', sections)
        self.assertNotIn('username', sections)

    def test_config_file_value(self):
        # test the set and get config file values
        # create the tmp config file path
        d = mkdtemp()
        f = join(d, 'config.txt')
        util.set_config_value('test1', 'val1', config_file_name=f)
        res = util.get_config_value('test1', config_file_name=f)
        self.assertEqual(res, 'val1')
        # test the fallback if a key doesn't exist
        res = util.get_config_value('test2', fallback='na', config_file_name=f)
        self.assertEqual(res, 'na')
        shutil.rmtree(d)

    def test_to_list(self):
        self.assertEqual(util._to_list(5), [5])
        self.assertEqual(util._to_list([5]), [5])
        self.assertEqual(util._to_list('test'), ['test'])
        self.assertEqual(util._to_list(range(5)), range(5))

    def test__argsort(self):
        vals = [1, 'C', 3.22, 'A', 1]
        idx = util._argsort(vals)
        self.assertEqual(idx, [0, 4, 2, 3, 1])


if __name__ == "__main__":
    main()
