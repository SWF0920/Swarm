# Makefile for "Who Gets to Ask for Help? Coordinating Human Oversight in Agent Swarms"

LATEX = pdflatex -interaction=nonstopmode -halt-on-error
PYTHON = python3

.PHONY: all pdf tikz data verify plots clean

all: pdf

pdf:
	$(LATEX) swarm_oversight_proposal.tex
	$(LATEX) swarm_oversight_proposal.tex

tikz:
	$(LATEX) fig_architecture.tex
	$(LATEX) fig_policies.tex
	$(LATEX) fig_specialization.tex

data:
	$(PYTHON) code/generate_tables_and_stats.py

verify:
	$(PYTHON) code/verify_mc_and_fd.py
	$(PYTHON) code/run_trust_trap_sweep.py --quick

plots:
	$(PYTHON) code/plot_figures.py

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg
