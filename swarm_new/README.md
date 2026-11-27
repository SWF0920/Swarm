# Who Gets to Ask for Help? Coordinating Human Oversight in Agent Swarms

Overleaf-ready LaTeX source, standalone TikZ figure sources, precomputed CSV data, and Python simulation/verification scripts for **Who Gets to Ask for Help? Coordinating Human Oversight in Agent Swarms** (Jiayuan Sheng and Mohamed El Amine Lakhnichi, Columbia University).

## Manuscript Structure

- **Main Paper (Pages 1–5, strict 5-page main-body limit)**:
  - Abstract & Section 1: Introduction
  - Section 2: Model Formulation (`fig_architecture.pdf`, `fig_policies.pdf`)
  - Section 3: Experimental Results — 3.1 Zero-false-alarm crowding out (`fig_crowding_vs_N.pdf`) and 3.2 Admission coordination via rotating rosters and decentralized IPG (`fig_specialization.pdf`)
  - Section 4: Conclusion (strategic deployment guidance)
- **References (Pages 6–7)**: 38 verified citations across AI control, multi-agent oversight, congested learning-to-defer, queueing, and team games.
- **Appendices (Pages 8–14)**:
  - Appendix A: Formal Propositions A.1–A.4 and proofs.
  - Appendix B: Baseline calibration, priority-noise and alarm-fatigue scaling laws (`fig_scaling_laws.pdf`), fixed-size rotating roster U-curves (`fig_optimal_roster.pdf`), passive-loss-of-control review volume (`fig_review_volume.pdf`), and single-asker trust traps (`fig_trust_trap_basin.pdf`).
  - Appendix C: Extended related work, structural comparison matrix (`Table 1`), and parameter-sharing / modeling scope discussion.

## Three-Role Semantic Color Palette (`colW`, `colF`, `colH`)

All vector PDF figures, TikZ sources, prose mentions (`\Wtxt`, `\Ftxt`, `\Htxt`), and mathematical variables (`\Wv`, `\Fv`, `\Hv`, `\oversee`, `\trust`, `\Over`) share a high-contrast, colorblind-distinct three-role palette:

- **Workers ($W$, `colW`)** — Cobalt Blue `RGB(25, 100, 186)` (`#1964BA`)
- **Frontier Orchestrator ($F$, `colF`)** — Royal Crimson-Plum `RGB(157, 23, 77)` (`#9D174D`)
- **Human Overseer ($H$, `colH`)** — Forest Emerald-Teal `RGB(15, 118, 94)` (`#0F765E`)

## Quick Start

1. **Compile the manuscript (Overleaf or local TeX Live):**
   ```bash
   make pdf
   # or directly:
   pdflatex swarm_oversight_proposal.tex && pdflatex swarm_oversight_proposal.tex
   ```
2. **Recompute exact CSV tables and verify analytical vs. Monte Carlo / FD checks:**
   ```bash
   python3 code/generate_tables_and_stats.py
   python3 code/verify_mc_and_fd.py
   python3 code/run_trust_trap_sweep.py --quick
   ```
3. **Recompile standalone TikZ schematics (`fig_architecture.pdf`, `fig_policies.pdf`, `fig_specialization.pdf`):**
   ```bash
   make tikz
   ```
