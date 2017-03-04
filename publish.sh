#!/bin/bash

git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"
# clear the old html if there is any
make -C doc clean
# clone the gh-pages branch so we can keep its commit history
git clone -b gh-pages --single-branch https://$GH_TOKEN@github.com/biocore/calour.git doc/_build/html
# update the html doc by build new doc
make -C doc html
# go to the html dir to commit changes and upload
cd doc/_build/html
git status
# add and commit all changes
git add *
git commit -m "$1"
git push --quiet origin gh-pages
