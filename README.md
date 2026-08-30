# GROW-CATALOGUE

Code accompanying the report **"No-Regret Learning in Stackelberg Security
Games with an Unknown, Bounded Set of Attacker Types"**.

The algorithm and regret bound are described in `gt_report.pdf`.
`project_report_extended.pdf` is that report verbatim plus four new sections
(9-12) and four appendices:

* an impossibility theorem for partial-information feedback with an unknown
  attacker set -- linear regret even with a single attacker type;
* an exact bit price for the side information that repairs it: on a universe
  of `M` candidate types, and given any oracle emitting `b` bits over the
  whole horizon, the optimal regret is `min{T, log2 M - b} +- 3`;
* the BLOCK-GROW-CATALOGUE algorithm, which recovers sublinear regret from a
  periodic set-revelation oracle, together with a matching lower bound in the
  revelation length and a separation between scheduled and defender-chosen
  report times;
* a sqrt(T) lower bound as soon as there are two attacker types;
* experiments: the lower bounds in simulation (Section 11), and tractable
  runs at n = 10 and n = 20 (Appendix D).

This repository reproduces every figure in both reports.

## Directory layout

    gt_report.pdf               compiled report (Sections 1-8)
    project_report.tex          report source
    project_report_extended.tex source of the extended report (Sections 1-12
                                plus Appendices A-D)
    project_report_extended.pdf compiled extended report
    refs.bib                    bibliography

    code/
      algorithm.py              core data structures and GROW-CATALOGUE
      plot_per_round_regret.py  Figure 1 (rolling per-round regret, n = 3)
      plot_regret_scaling.py    Figure 2 (Regret vs T scaling, n = 3)

      oracle.py                 the best-response oracle of Appendix C:
                                linear optimisation over the lifted set,
                                by profile enumeration (n^|C| LPs) or by one
                                MILP; replaces the explicit set E(C; eps)
      tractable.py              Oracle-Grow-Catalogue (Algorithm 1) and
                                Block-Grow-Catalogue (Algorithm 2), each one
                                oracle call per round
      bench_oracle.py           Table 3 (enumeration vs oracle runtimes; App. C)
      plot_tractable_regret.py  Figure 4 (Algorithm 1 at n = 10, 20; App. D)
      plot_block_revelation.py  Figure 5 (Algorithm 2 at n = 10, 20; App. D)
      plot_impossibility.py     Figure 3 (Theorems 12, 15, 24 and
                                Propositions 16, 26 in simulation)
      verify_theory.py          re-runs the non-asymptotic claims of
                                Sections 9-10 and Appendix C; prints PASS/FAIL

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

    # Sections 1-8
    python plot_per_round_regret.py     # -> ../figures/avg_regret_per_iter.{pdf,png}
    python plot_regret_scaling.py       # -> ../figures/regret_vs_T_scaling.{pdf,png}

    # Sections 9-12
    python verify_theory.py             # -> PASS/FAIL for every claim
    python bench_oracle.py              # -> Table 3, printed
    python plot_impossibility.py        # -> ../figures/impossibility.{pdf,png}
    python plot_tractable_regret.py     # -> ../figures/tractable_regret.{pdf,png}
    python plot_block_revelation.py     # -> ../figures/block_revelation.{pdf,png}

Each script averages over many random games and prints per-run diagnostics.
`verify_theory.py` and `plot_impossibility.py` run in a couple of minutes;
the two regret scripts fan their seeds out over `multiprocessing` and take
roughly an hour of wall-clock each on a 24-core machine. Both accept
`--fast` for a short smoke test and `--replot` to redraw from the cached
curves in `figures/*_curves.npz` without rerunning the simulations.
`verify_theory.py` also accepts `--quick`.

## Rebuilding the reports

    pdflatex project_report.tex
    bibtex   project_report
    pdflatex project_report.tex
    pdflatex project_report.tex
    mv project_report.pdf gt_report.pdf

    pdflatex project_report_extended.tex
    bibtex   project_report_extended
    pdflatex project_report_extended.tex
    pdflatex project_report_extended.tex

Sections 1-8 of `project_report_extended.tex` are byte-identical to
`project_report.tex`; Sections 9-12 are appended before the bibliography and
Appendices A-D after it.

Requires a TeX Live installation with `texlive-latex-recommended`,
`texlive-latex-extra`, and `texlive-science` (the last supplies
`algorithm.sty` and `algpseudocode.sty`).
