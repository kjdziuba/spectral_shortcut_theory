# Lab talk — Spectral Shortcut Theory

7-slide deck for the lab meeting.

## Files

- `build_lab_talk.py` — Python script that generates the PPTX. Re-run
  to regenerate after any plot or numbers change.
- `spectral_shortcut_lab_talk.pptx` — the generated deck.

## Slide map

1. **Title** — left-block layout, project name + subtitle.
2. **The empirical paradox** — table of 7 variants from the companion
   empirical paper (learned 0.675 → frozen Slidewin 0.95).
3. **Setup** — pipeline diagram (input → f_θ → g_φ → ŷ), capacity
   asymmetry C_f vs C_g, the framed question.
4. **Chain rule decomposition** — ∂L/∂θ = ∂L/∂ŷ · ∂ŷ/∂Z · ∂Z/∂θ
   with three colored columns explaining each term and what dies first.
5. **Shortcut analogy** — Pezeshki's two-moons on the left, our
   chemistry-vs-morphology on the right, plus the structural takeaway.
6. **Preliminary results** — Exp 1.1 paired eigenvalues (κ blowup)
   on the left, Exp 1.3 killer test bar chart on the right, with
   bottom captions explaining the synthetic data setup.
7. **Where this goes** — paper deliverables (Thms 1+2, EGR, real-data
   link) and the practical prescription (freeze, residual baseline,
   track EGR).

## How to update

- Numbers/plots: just edit the script, re-run.
- Style: change the constants at the top (NAVY, BLUE, ORANGE, …).
- Re-order slides: each slide is self-contained between two
  `# Slide N` comments — move the blocks.

## Re-build

```
python presentation/build_lab_talk.py
```
