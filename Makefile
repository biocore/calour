# ----------------------------------------------------------------------------
# Copyright (c) 2015--, calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

.DEFAULT_GOAL := help

ifeq ($(WITH_COVERAGE), TRUE)
	TEST_COMMAND = COVERAGE_FILE=.coverage coverage run --rcfile .coveragerc setup.py nosetests --with-doctest
else
	TEST_COMMAND = nosetests --with-doctest
endif

CMSG ?= update html doc

help:
	@echo 'Use "make test" to run all the unit tests and docstring tests.'
	@echo 'Use "make pep8" to validate PEP8 compliance.'
	@echo 'Use "make html" to create html documentation with sphinx'
	@echo 'Use "make all" to run all the targets listed above.'
	@echo 'Use "make doc_upload" to create html documentation and upload to github pages.'
test:
	$(TEST_COMMAND)
pep8:
	flake8 calour setup.py
html:
	make -C doc clean html
all: test pep8 html
doc_upload:
	make -C doc clean
	cd doc/_build/ && git clone -b gh-pages --single-branch git@github.com:biocore/calour.git html
	make -C doc html
	cd doc/_build/html && git commit -a -m "$(CMSG)" && git push origin gh-pages
