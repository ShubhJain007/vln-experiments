# Technical report

**Evaluating Different Approaches to Vision-and-Language Navigation: An Experimental Study on a Single GPU**:
[`main.pdf`](main.pdf) (19 pages).

| File | What it is |
|---|---|
| `main.tex` | LaTeX source (article class, natbib, TikZ) |
| `references.bib` | all 53 references; arXiv entries exported from arxiv.org |
| `figures/` | copies of `../docs/figures/*.png`, which `../tools/make_figures.py` regenerates from `../logs/` |
| `Makefile` | `make` builds `main.pdf`; `make arxiv` builds an upload bundle; `make figures` refreshes the figures |

Building needs [tectonic](https://tectonic-typesetting.github.io) (a single binary that fetches LaTeX packages on
demand), or any TeX Live with `latexmk -xelatex main.tex`.

**Licence.** The text is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Figures showing
rendered Matterport3D scenes (`fig_qualitative.png`, `fig_reasoner_zeroshot.png`) are for non-commercial academic use
only, under the [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).
