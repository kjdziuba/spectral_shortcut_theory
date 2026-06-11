# Spectral Shortcut Theory

Theoretical paper on why end-to-end joint training fails in compositional spectral-spatial deep learning pipelines.

**Working title**: *The Spectral Shortcut Theorem: A Gradient-Geometric Explanation for Failure Modes of Joint Spectral-Spatial Deep Learning*

**Target venue**: ICML / NeurIPS theory track (primary), AISTATS / IEEE TMI / Medical Image Analysis (backup)

**Timeline**: 6 months part-time, starting 2026-06-08

---

## What this paper proves

Two theorems explaining the empirical observation that frozen spectral encoders dramatically outperform end-to-end joint training in spectral-spatial pipelines (e.g., IR/QCL tissue classification, hyperspectral remote sensing).

### Theorem 1 — Hessian Capacity Bound
The condition number of the joint Hessian κ(H) scales as Ω(C_g / C_f), where C_g and C_f are the parameter counts of the spatial and spectral modules. Capacity asymmetry produces geometric instability.

### Theorem 2 — Two-Timescale Spurious Convergence
The capacity imbalance induces a slow-fast dynamical system. The fast spatial module saturates cross-entropy loss using spatial shortcuts before the slow spectral module receives meaningful gradient signal. Training halts at a spurious stationary point.

### The diagnostic
**Effective Gradient Ratio (EGR)** = ‖∇_θ L‖ / ‖∇_φ L‖ — a real-time training-time metric that detects the onset of gradient starvation. Branded contribution, intended for PyPI release.

### The prescription
Freeze the spectral module (strong solution) or inject a fixed residual baseline (soft solution).

---

## Mathematical pillars

1. **Block Hessian analysis** — Schur complement bounds, Karakida mean-field FIM, Pennington-Bahri random matrix theory
2. **Two-timescale stochastic approximation** — Borkar 1997, Heusel 2017 (TTUR for GANs)
3. **Gradient starvation** — Pezeshki 2021, NTK-regime cross-entropy dynamics

---

## Directory structure

```
spectral_shortcut_theory/
├── README.md                   This file
├── PLAN.md                     Full 8-phase project plan
├── PROGRESS.md                 Running session log
├── paper/                      LaTeX project
│   ├── main.tex                Top-level document
│   ├── sections/               One file per section
│   ├── figures/                Final PDFs
│   ├── figures_src/            Source notebooks / scripts
│   └── references.bib
├── proofs/                     Working markdown proofs (drafted before LaTeX)
├── reading_notes/              Phase 0 deliverables (5 foundational papers)
├── code/                       Source code
│   ├── egr/                    The diagnostic implementation
│   ├── synthetic/              Toy experiment generators
│   ├── hessian/                Power iteration, eigenvalue tracking
│   └── experiments/            Training scripts and runners
├── configs/                    YAML configs for experiments
└── results/                    Outputs from runs
```

---

## How to navigate this project

- **First time here?** Read `PLAN.md` end-to-end.
- **Resuming a session?** Read the last entry in `PROGRESS.md`.
- **Working on proofs?** Drafts live in `proofs/*.md`, final LaTeX in `paper/sections/`.
- **Working on experiments?** Configs in `configs/`, runners in `code/experiments/`.

---

## Sibling projects

- **`../spectral_tokenization/`** — empirical paper (Paper 1: tokenization benchmark) and the empirical companion paper (Paper 2: val-vs-test methodology critique on real FTIR/QCL data)
- This theory paper is intentionally separated. It proves the mechanism; the empirical papers document the symptoms.
