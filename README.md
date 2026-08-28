# GROW-CATALOGUE

Code accompanying the report **"No-Regret Learning in Stackelberg Security
Games with an Unknown, Bounded Set of Attacker Types"**.

The algorithm and regret bound are described in `gt_report.pdf`. This
repository reproduces the two experimental figures in that report.

## Directory layout

    gt_report.pdf               compiled report (main deliverable)
    project_report.tex          report source
    project_report_extended.tex report source + Section 9 "Extensions"
    project_report_extended.pdf compiled extended report
    refs.bib                    bibliography

    code/
      algorithm.py              core data structures and GROW-CATALOGUE
      plot_per_round_regret.py  generates Figure 1 (rolling per-round regret)
      plot_regret_scaling.py    generates Figure 2 (Regret vs T scaling)

    figures/                    figures included in the report
    presentation/               slide deck and its own figures
    reference-papers/           Important relevant references

## Setup

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Reproducing the figures

From the `code/` directory:

    cd code
    python plot_per_round_regret.py     # -> ../figures/avg_regret_per_iter.{pdf,png}
    python plot_regret_scaling.py       # -> ../figures/regret_vs_T_scaling.{pdf,png}

Each script averages over many random games and prints per-run diagnostics.
Wall-clock is a few minutes on a laptop.

## Rebuilding the report

    pdflatex project_report.tex
    bibtex   project_report
    pdflatex project_report.tex
    pdflatex project_report.tex
    mv project_report.pdf gt_report.pdf

Same sequence for `project_report_extended.tex`, which is
`project_report.tex` verbatim plus a new Section 9 covering two
extensions (periodic type revelation; side information with an unknown
type set) and an erratum for Step 7 of the Theorem 6 proof.

Requires a TeX Live installation with `texlive-latex-recommended`,
`texlive-latex-extra`, and `texlive-science`.
