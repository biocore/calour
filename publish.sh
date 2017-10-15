#!/bin/bash
if [ "${TRAVIS_SECURE_ENV_VARS}" == "true" ] && [ "${CONDA_PY}" == "35" ] && [ "${TRAVIS_OS_NAME}" == "linux" ] ; then
    echo "Publish sphinx documentation"
    git config --global user.email "travis@foo.com"
    git config --global user.name "travis"
    # clear the old html if there is any
    make -C doc clean
    # clone the gh-pages branch so we can keep its commit history
    git clone -b gh-pages --single-branch https://$GH_TOKEN@github.com/biocore/calour.git doc/build/html
    # update the html doc by build new doc
    make -C doc html
    # go to the html dir to commit changes and upload
    cd doc/build/html
    git status
    # add and commit all changes
    git add *
    git commit -m "${TRAVIS_COMMIT_MESSAGE}"
    # use --quiet to not print private info
    git push --quiet origin gh-pages
else
    echo "Not publish sphinx documentation"
fi
