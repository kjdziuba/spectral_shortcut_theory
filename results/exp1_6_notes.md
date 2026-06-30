# Experiment 1.6 — Spectral Decoupling comparison

**Run**: 2026-06-30. 120 runs at WIDTH=256, EPOCHS=100, weight_decay=0.
**Sweep**: `joint`, `frozen`, `sd × λ ∈ {1e-3, 1e-2, 1e-1, 1.0}` × noise ∈ {0.05, 0.10, 0.15, 0.20} × 5 seeds.
**SD form**: `(λ/2) · ||logits||²` on pre-softmax logits, sum-over-classes / mean-over-examples (Pezeshki et al. 2021, Eq. 8).

## Headline (aggregated over 5 seeds, 4 noise levels)

| Condition       | Peak acc | Final acc | Peak − final (collapse) |
|-----------------|---------:|----------:|------------------------:|
| joint           |    0.660 |     0.547 |               **0.111** |
| SD (best λ/n)   |    0.660 |     0.550 |               **0.110** |
| frozen          |    0.583 |     0.517 |               **0.066** |

Paired Wilcoxon (matched on (noise, seed), n=20):
- **joint vs SD(best λ)**: T = 88, p = 0.55 → not significant
- **joint vs frozen**: T = 6, **p < 0.0001** → frozen reduces collapse by ~41%

## Interpretation

Pezeshki's spectral decoupling does not significantly reduce the peak-to-final test-accuracy collapse our two-timescale theory predicts. Freezing — the prescription Theorem 2 directly motivates — does.

This is the expected result given the *mechanism* of each intervention:

- SD penalises logit magnitude, which slows cross-entropy saturation but does not eliminate the timescale gap between θ and φ.
- Freezing collapses the timescale gap to zero by removing θ from the optimisation entirely.

The result strengthens the paper's positioning: SD addresses a different failure mode (gradient starvation in shared-representation classifiers) than what our theory targets (compositional timescale collapse in a stacked spectral→spatial pipeline).

## Caveats (be honest in the paper)

1. **Frozen trades off peak**: at every noise level, frozen has lower peak test accuracy
   (0.583 vs 0.660) because the random spectral projection sometimes destroys signal.
2. **Frozen variance is high**: σ_seed = 0.08–0.14 for frozen vs 0.02–0.04 for joint/SD.
   A single seed at noise=0.20 (seed=46) frozen achieved acc=0.372 (worse than chance).
   The fixed-residual baseline alternative (Section 6) avoids this lottery.
3. **At D=256, joint training does not catastrophically collapse**: peak 0.66 → final 0.55 =
   11-point drop. Exp 1.2 v3 showed bigger collapses at D=2048. Exp 1.6 is in the moderate-
   capacity regime; the SD-vs-freeze comparison is sharpest there.
4. **Best SD λ depends on noise**: λ=0.01 best at σ=0.05; λ=0.10 best at σ≥0.10. A single λ
   across noise levels is suboptimal — SD requires per-task tuning that freezing does not.

## Where this lands in the paper

Insert as a new subsection in **Section 7 (Experiments)** before the discussion. Headline figure:
[exp1_6_collapse_comparison.png](exp1_6_collapse_comparison.png) (peak vs final + collapse-by-condition bars).
Supplementary figure: [exp1_6_lambda_sweep.png](exp1_6_lambda_sweep.png) (gap vs λ per noise).

Reviewer ask "did you compare to Pezeshki's spectral decoupling?" is now preempted.

## Artifacts

- `results/exp1_6_summary.csv` — 120 rows × 14 columns
- `results/exp1_6/egr/` — per-run EGR trajectories
- `results/exp1_6/metrics/` — per-run epoch-level metrics
- `results/exp1_6_collapse_comparison.png` — headline
- `results/exp1_6_lambda_sweep.png` — λ sweep supplement
- `code/experiments/exp1_6.py` — runner
- `logs/exp1_6.log` — stdout

## Reproducing

```bash
cd /home/u37314kd/Projects/spectral_shortcut_theory
python code/experiments/exp1_6.py
```

Wall time: ~5 min on RTX 5000 Ada (synthetic problem is small).
