# kandy-skills

Open-source [Claude Code skills](https://code.claude.com/docs/en/skills) for
[KANDy](https://github.com/Center-For-Complex-Systems-Science/kandy) —
Kolmogorov-Arnold Networks for Dynamics, a scientific Python library for
data-driven identification of dynamical systems (`x_dot = A · Ψ(φ(x))`).

These skills teach coding agents how to use KANDy correctly: choosing Koopman
lifts, training single-layer KANs, extracting symbolic governing equations, and
generating finite-volume PDE training data — with runnable examples for a dozen
benchmark systems.

## Skills

| Skill | Description |
|---|---|
| [`kandy`](kandy/SKILL.md) | System identification / equation discovery with KANDy: lift selection, training, symbolic extraction, FV numerics, and 18 example scripts |

## Installation

Copy the skill directory into your project's or user's skills folder:

```bash
# Project-level (shared with collaborators via git)
mkdir -p .claude/skills
cp -r kandy .claude/skills/

# Or user-level (available in all your projects)
mkdir -p ~/.claude/skills
cp -r kandy ~/.claude/skills/
```

Claude Code discovers the skill automatically from its `SKILL.md` frontmatter
and loads the reference docs and examples on demand.

## Layout

```
kandy/
├── SKILL.md              # entry point: core workflow + when to use what
├── references/
│   ├── lifts.md          # Koopman lift selection guide
│   ├── training.md       # optimizers, rollout loss, discrete maps, periodic phases
│   ├── symbolic.md       # symbolic extraction, custom libraries, scoring, LaTeX
│   ├── numerics.md       # finite-volume PDE data generation
│   └── api.md            # full public API reference
└── examples/             # 18 complete, runnable scripts
    ├── lorenz_example.py
    ├── henon_example.py
    ├── ikeda_example.py
    ├── kuramoto_example.py
    ├── burgers_example.py
    ├── kuramoto_sivashinsky_example.py
    ├── navier_stokes_example.py
    └── ...
```

## Requirements

The skill assumes the `kandy` package is installed in the working environment:

```bash
pip install kandy
```

Python 3.11–3.13 · PyTorch ≥ 2.0 · PyKAN ≥ 0.2.0 · SciPy ≥ 1.10 · SymPy ≥ 1.12

## License

MIT — see [LICENSE](LICENSE).
