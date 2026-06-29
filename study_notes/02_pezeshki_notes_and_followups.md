# Pezeshki — what to revisit + open follow-ups

## What we covered (session 2026-06-29)

- **Spectral decoupling** (their fix): L2 penalty on **logits**, not weights.
  Loss = CE(ŷ, y) + (λ/2)·‖ŷ‖². Bounds logits away from ±∞ so cross-entropy
  doesn't fully saturate, and even weak features keep getting gradient.
- **The two-moons-style example**: GD picks the linear separator over the
  curved one even when curved is more robust. Maps to our spectroscopy
  setup: "linear separator" = spatial morphological shortcut,
  "curved boundary" = the chemistry we want the model to learn.
- **Robustness claim**: gradient starvation is NOT cured by training
  longer, weight decay, dropout, batch norm, Adam vs SGD, architecture
  change, or coordinate change. Pezeshki Sec. 4 + App B.

## TODOs from this session

1. **Add Spectral Decoupling as an experimental condition** (somewhere
   in benchmarks / Exp 1.5 or new Exp 1.6). The condition: same CNN,
   same data, but training with the CE + λ‖ŷ‖² loss instead of plain CE.
   Predicted: helps but doesn't match freezing, because our pathology is
   structural (module-level timescale) not just CE saturation.

2. **Cite Pezeshki Sec 4 / App B robustness claim** in our Discussion
   (Section 9) to avoid having to re-run those ablations ourselves.

3. **Prepare 3-4 slide presentation** for the team meeting. Audience:
   spectroscopy people, math newbies. The two-moons analogy should be
   on a slide. See `study_notes/03_team_slides.md` (to be written).

## What to re-read in Pezeshki — iteratively over the next few sessions

Schedule we agreed on: don't try to absorb the whole paper at once; come
back to it in small bites as the project progresses.

- **Pass 2** (next): Sec 4 in detail — exact mechanism of spectral
  decoupling, how λ scales with feature strength. Goal: be able to
  write the implementation in one paragraph.
- **Pass 3** (during Theorem 2 proof): Sec 3 NTK proof in detail —
  the eigenvalue decay argument for the residual r_k(t). Goal:
  re-derive their Eq. 3.7 (or whichever bounds the residual).
- **Pass 4** (before submission): App B — robustness claims with
  exact experimental detail so we can cite the right page numbers.

## Open question (parked): do we need NTK measurements ourselves?

Our framework uses **Hessian / Gauss-Newton eigenvalues** as the
primary observable (Theorem 1, EGR diagnostic, Exp 1.1). These are
related to NTK eigenvalues near minima (the Gauss-Newton matrix is
asymptotically equal to the empirical NTK), but they are not identical
in the finite-data, finite-width, mid-training regime where our
experiments live.

NTK measurements would tighten the Pezeshki-connection in Theorem 2's
proof, where we adapt his feature-level argument to our module-level
setting. But for the theorems' core statements — particularly
Theorem 1's κ(G) ~ Ω(Cg/Cf) — Hessian eigenvalues are the right tool.

Decision (parked, can revisit): **no NTK measurements in main text.**
We may want one supplementary figure showing the empirical NTK spectrum
for the spectroscopy CNN to make the Pezeshki connection more explicit,
but it's not load-bearing.
