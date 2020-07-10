# ----------------------------------------------------------------------------
# Copyright (c) 2015--, calour development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

.DEFAULT_GOAL := help

ifeq ($(WITH_COVERAGE), TRUE)
	TEST_COMMAND = pytest -svv --doctest-modules --doctest-continue-on-failure --cov calour --cov-report term-missing --cov-report html:cov_html
else
	TEST_COMMAND = pytest -svv --doctest-modules --doctest-continue-on-failure calour
endif


MSG ?=


help:
	@echo 'Use "make test" to run all the unit tests and docstring tests.'
	@echo 'Use "make pep8" to validate PEP8 compliance.'
	@echo 'Use "make html" to create html documentation with sphinx'
	@echo 'Use "make all" to run all the targets listed above.'
	@echo 'Use "MSG=whatever_update_msg make publish" to create html documentation and upload to github pages.'
test:
	$(TEST_COMMAND)
pep8:
	flake8 calour setup.py
html:
	make -C doc clean html
publish:
# force to specify a nonempty git commit msg
ifeq ($(MSG), )
	@echo 'You need to specify a git commit msg to update and publish doc! For example:'
	@echo 'MSG="update doc to version 1.0" make publish'
else
	make -C doc clean
	git clone -b gh-pages --single-branch git@github.com:biocore/calour.git doc/build/html
	make -C doc html
	cd doc/build/html && git add * && git commit -m "$(MSG)" && git push origin gh-pages
endif



all: test pep8 html

