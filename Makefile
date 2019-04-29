MARKDOWN_FILES = $(wildcard assignment*/*.md)
NOTEBOOK_FILES = $(addsuffix .ipynb, $(basename $(MARKDOWN_FILES)))

%.ipynb: %.md
	$(info Found markdown files: [${MARKDOWN_FILES}])
	$(info Building notebooks: [${NOTEBOOK_FILES}])
	jupytext --to notebook $<
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks: $(NOTEBOOK_FILES)

clean:
	@rm -f $(NOTEBOOK_FILES)