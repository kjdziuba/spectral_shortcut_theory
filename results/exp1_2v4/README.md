# Exp 1.2 v4 — per-run artifacts and claim provenance

## Metric backing the ViT universality claim (audit A60)

The claim that the joint-vs-frozen dissociation is NOT CNN-specific is
backed by **final-epoch test accuracy** (`final_test_acc` in
`../exp1_2v4_summary.csv`), mean over seeds {42, 43, 44}, comparing
condition `frozen` vs `joint` at each ViT width — the SAME metric used
for the CNN. Peak test accuracy (`peak_test_acc`) is reported as a
secondary check. The EGR time series (`egr_*.csv`, column `egr` =
||grad_theta|| / ||grad_phi||, logged every step after backward) are
mechanistic support for the two-timescale interpretation, not the
headline claim.

## Files

- `egr_{arch}_{cond}_D{width}_s{seed}.csv` — per-step EGR:
  `step, grad_theta_norm, grad_phi_norm, egr`. In the `frozen`
  condition `grad_theta_norm` is 0 by construction (requires_grad=False).
- `metrics_{arch}_{cond}_D{width}_s{seed}.csv` — per-epoch:
  `epoch, train_loss, test_loss, test_acc`.
- `../exp1_2v4_summary.csv` — one row per run (final/peak metrics).

## Joint-vs-frozen gaps (mean over seeds; gap = frozen - joint)

| arch | width | final_acc joint | final_acc frozen | gap (final) | peak_acc joint | peak_acc frozen | gap (peak) |
|------|-------|-----------------|------------------|-------------|----------------|-----------------|------------|
| cnn | 16 | 0.6178 | 0.6442 | +0.0264 | 0.6823 | 0.6876 | +0.0054 |
| cnn | 64 | 0.5737 | 0.6114 | +0.0377 | 0.7032 | 0.6881 | -0.0151 |
| cnn | 256 | 0.5399 | 0.5736 | +0.0337 | 0.7116 | 0.6896 | -0.0220 |
| cnn | 1024 | 0.5336 | 0.5597 | +0.0262 | 0.7107 | 0.6896 | -0.0210 |
| vit | 64 | 0.5331 | 0.5473 | +0.0142 | 0.7072 | 0.7016 | -0.0055 |
| vit | 128 | 0.5464 | 0.5669 | +0.0205 | 0.7060 | 0.6786 | -0.0275 |
| vit | 256 | 0.5028 | 0.5393 | +0.0365 | 0.7071 | 0.7001 | -0.0070 |

Setup: S=64, K=16, H=W=16, 2 classes, N_train=512,
N_test=128, batch=32, 150 epochs, Adam lr=0.001,
data noise=0.1. Regenerate everything with:
`python code/experiments/exp1_2v4.py`.
