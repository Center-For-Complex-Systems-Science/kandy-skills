# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Claude Code **skill/plugin repository** — documentation and example scripts that teach coding agents how to use [KANDy](https://github.com/Center-For-Complex-Systems-Science/kandy) (Kolmogorov-Arnold Networks for Dynamics), a Python library for data-driven discovery of dynamical systems (`x_dot = A · Ψ(φ(x))`). The KANDy library itself lives elsewhere; this repo contains no importable package, build system, or test suite.

## Structure

- `.claude-plugin/plugin.json` — plugin manifest; `marketplace.json` makes the repo installable as its own marketplace (`/plugin install kandy@kandy-skills`).
- `skills/kandy/SKILL.md` — the skill entry point. Its frontmatter `description` drives skill discovery/triggering; the body is the always-loaded core (workflow, lift-selection table, example index).
- `skills/kandy/references/` — on-demand deep-dives loaded from SKILL.md: `lifts.md`, `training.md`, `symbolic.md`, `numerics.md`, `api.md`.
- `skills/kandy/examples/` — self-contained runnable scripts, one per benchmark system, grouped into subdirectories by fit recipe (`odes/`, `maps/`, `oscillators/`, `pdes/`, `fluids/`, `mathbio/`, `geometry/`). Each simulates its own training data, fits a KANDy model, validates by rollout, and extracts symbolic formulas. New examples go in the category matching their data type / training technique — the categories exist so an agent can find the recipe to imitate, so keep one obvious example per technique.

## Working on the skill

- **Progressive disclosure is the design**: SKILL.md stays short and points into `references/` and `examples/`. Put detail in a reference file, not SKILL.md.
- **Three tables must stay in sync with the files on disk**: the example tables in `SKILL.md`, `examples/README.md`, and the layout section of the top-level `README.md`. When adding, renaming, or deleting an example or reference file, update all of them.
- The core domain rule the skill exists to teach (worth preserving in any rewrite): the KAN is separable, so cross-interaction terms must be encoded explicitly in the lift φ — a missing cross-term is a structural error, not a tuning issue.

## Running examples

Examples require the KANDy library installed in the environment (`pip install kandy`; Python 3.11–3.13, PyTorch ≥ 2.0, PyKAN ≥ 0.2.0):

```bash
cd skills/kandy/examples
python odes/lorenz_example.py
```

Scripts use the matplotlib `Agg` backend and write figures to the working directory — no display needed. There is no test suite; validating a changed example means running it and checking the rollout/figures/R² it reports.
