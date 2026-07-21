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
| [`kandy`](skills/kandy/SKILL.md) | System identification / equation discovery with KANDy: lift selection, training, symbolic extraction, FV numerics, and example scripts |

## Installation

### As a plugin (recommended)

This repo is a Claude Code plugin and its own marketplace. In Claude Code:

```
/plugin marketplace add Center-For-Complex-Systems-Science/kandy-skills
/plugin install kandy@kandy-skills
```

Or from the terminal:

```bash
claude plugin marketplace add Center-For-Complex-Systems-Science/kandy-skills
claude plugin install kandy@kandy-skills
```

### Manual copy

Alternatively, copy the skill directory into a skills folder:

```bash
# Project-level (shared with collaborators via git)
mkdir -p .claude/skills
cp -r skills/kandy .claude/skills/

# Or user-level (available in all your projects)
mkdir -p ~/.claude/skills
cp -r skills/kandy ~/.claude/skills/
```

Claude Code discovers the skill automatically from its `SKILL.md` frontmatter
and loads the reference docs and examples on demand.

## Layout

```
.claude-plugin/
├── plugin.json           # Claude Code plugin manifest
└── marketplace.json      # lets the repo be added as a plugin marketplace
skills/kandy/
├── SKILL.md              # entry point: core workflow + when to use what
├── references/
│   ├── lifts.md          # Koopman lift selection guide
│   ├── training.md       # optimizers, rollout loss, discrete maps, periodic phases
│   ├── symbolic.md       # symbolic extraction, custom libraries, scoring, LaTeX
│   ├── numerics.md       # finite-volume PDE data generation
│   └── api.md            # full public API reference
└── examples/             # complete, runnable scripts, grouped by fit recipe
    ├── odes/             # continuous ODEs + delay DEs (Lorenz, Van der Pol, Mackey–Glass, …)
    ├── maps/             # discrete maps (Hénon, Ikeda)
    ├── oscillators/      # coupled phase oscillators (Kuramoto, network reconstruction, …)
    ├── pdes/             # 1D PDEs (Burgers, Kuramoto–Sivashinsky)
    ├── fluids/           # fluid dynamics (vortex shedding, 2D turbulence, 3D Navier–Stokes)
    ├── mathbio/          # mathematical biology (SIR, Lotka–Volterra, Epileptor, …)
    └── geometry/         # engineered lifts on manifolds and Lie groups
                          #   (Hopf, trefoil, Euler top, Heisenberg, SE(2), …)
```

## Requirements

The skill assumes the `kandy` package is installed in the working environment:

```bash
pip install kandy
```

Python 3.11–3.13 · PyTorch ≥ 2.0 · PyKAN ≥ 0.2.0 · SciPy ≥ 1.10 · SymPy ≥ 1.12

## License

MIT — see [LICENSE](LICENSE).
