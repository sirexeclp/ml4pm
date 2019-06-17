#.ONESHELL:
MARKDOWN_FILES = $(wildcard assignment*/*.md)
NOTEBOOK_FILES = $(addsuffix .ipynb, $(basename $(MARKDOWN_FILES)))
#RENAMED_NOTEBOOK_FILES = $(subst STUDENT,HierKönnteIhrGruppennameStehen, $(NOTEBOOK_FILES))

%.ipynb: %.md
	$(info Found markdown files: [${MARKDOWN_FILES}])
	$(info Building notebooks: [${NOTEBOOK_FILES}])
	$(info This notebook will be renamed to: $(subst STUDENT, HierKönnteIhrGruppennameStehen, $@))
	jupytext --to notebook $<
	jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute --inplace $@
	mv $@ $(subst STUDENT,HierKönnteIhrGruppennameStehen, $@)

notebooks: $(NOTEBOOK_FILES)
#	touch $@

clean:
	#remove notebooks and checkpoints
	@rm -rf */*.ipynb
	@rm -rf */.ipynb_checkpoints