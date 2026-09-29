# Build the paper:  make -C paper      (needs tectonic: https://tectonic-typesetting.github.io)
# Figures are copies of docs/figures/*.png, which tools/make_figures.py regenerates from logs/.
main.pdf: main.tex references.bib figures/*.png
	tectonic -X compile main.tex

figures:
	cp ../docs/figures/*.png figures/

arxiv: main.pdf
	tectonic -X compile main.tex --keep-intermediates
	tar czf arxiv.tar.gz main.tex main.bbl references.bib figures

.PHONY: figures arxiv
