# ml-in-prec-med
[![Build Status](https://travis-ci.com/sirexeclp/ml-in-prec-med.svg?branch=master)](https://travis-ci.com/sirexeclp/ml-in-prec-med)
## Workflow

Work on local your branch (e.g. `your-name`).
Before you start make sure to `merge upstream master` into your branch :).

When youre good to go: open a pull request from your branch to master.

After your PR got merged, travis will build and execute the notebook.
The result is pushed into the `build` branch.

Please **do not** change files in the build branch!

`build` should only contain files that have been build by travis.

Instead make changes on master or even better your branch and pr.

## Local Setup

If you have the jupytext extension installed, you can start jupyter as usuall `jupyter notebook` and just open the markdown-file for the exercise.

Please do not commit notebooks (`.ipynb`)!
Instead commit the markdown version only.

`.ipynb` notebooks should be `.gitignored`.

If you need to manually convert notebooks you can 
generate a notebook with 

    jupytext --to notebook <file.md>

and convert it back to markdown with

    jupytext --to markdown <file.ipynb>

## Install jupytext

Instructions on how to install jupytext can be found [here](https://github.com/mwouts/jupytext).

Or just copy this:

    pip install jupytext --upgrade
    jupyter notebook --generate-config
    echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> .jupyter/jupyter_notebook_config.py
    jupyter nbextension install --py jupytext --user
    jupyter nbextension enable --py jupytext --user

## Instructions

Welcome to the first tutorial of the Machine Learning course! 

Because all of you come from different disciplines, we want to make sure that we create a common baseline for all of us. Those of you who are already experienced with python and basic statistics, will find this tutorial too easy. 

At first, please make sure that you can all access jupyter-lab: Go through the following chart and 

1. Install anaconda: Source
2. Open Anaconda → Click on Jupyter notebook (second icon)  → A shell and browser window opens
3. Download the provided jupyter notebook for this tutorial and save it on your local computer.
4. You see your local directories in the jupyter browser tab. Navigate to the folder where you saved the tutorials jupyter notebook (Step 3) and open this notebook
5. Great, now we are ready to start coding!
6. Download diabetes dataset: Link
