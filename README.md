# ml-in-prec-med

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