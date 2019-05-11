#!/bin/bash
git config --global user.email ${EMAIL}
git config --global user.name ${USER}
git checkout build
git add *.ipynb
git commit -m "[skip travis] Travis build: ${TRAVIS_BUILD_NUMBER}"
git remote add origin-pages https://${TOKEN}@github.com/sirexeclp/ml-in-prec-med.git > /dev/null 2>&1
git push --quiet --set-upstream origin-pages build