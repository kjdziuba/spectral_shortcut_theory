# ICLR 2027 main text — skeleton v3 (refocused paper; 2026-09-10 20:30)

Spec for `paper/main_iclr_v2.tex` + `paper/sections_iclr_v2/*.tex` (NEW
files; v1 = `main_iclr.tex`/`sections_iclr/` stays frozen at tag
`iclr-v1-2026-09-10`; v1 spec = `ICLR_SKELETON.md`). Source of the
decisions: `REFOCUS_PLAN_2026-09-10.md` (read it first) and Astra's
`review_packet/astra/review_iclr_01.md`. Placeholders `[E2]`, `[E3]` are
filled from `results/exp2_summary.csv` / `results/exp3_*` only.

## 0. The claim (spine of abstract, intro, conclusion)

> In a serial spectral–spatial model, increasing the effective training
> speed of the contextual pathway suppresses learning of an equally
> predictive spectral cue. At matched training fit, this produces greater
> reliance on context and failure when that context changes.

Three levels, always in this order: what we PROVE (serial linear-encoder CE
model), what we PREDICT (any serial model whose contextual pathway learns
fast relative to the encoder), what we OBSERVE (Exp 2 nonlinear synthetic,
Exp 3 real spectra) — each with a control that removes the cause.

Title (working): "The Spectral Shortcut: How a Fast Contextual Pathway
Suppresses Spectral Learning in Serial Spectral–Spatial Models".

## 1. Page budget (9.00 pp incl. figures; ≈ 4,300 words of prose)

| # | Section (file in `sections_iclr_v2/`) | Pages | Words | Displays | Figures/tables |
|---|---|---|---|---|---|
| — | Title + abstract (`00_abstract.tex`) | 0.30 | 160 | — | — |
| 1 | Introduction with focused related work (`01_intro.tex`) | 1.20 | 750 | 0 | — |
| 2 | Model, task, definition of failure (`02_model.tex`) | 0.75 | 380 | 2 | **Fig 1** (model + cues + tests) |
| 3 | Main theorem and explanation (`03_theorem.tex`) | 2.00 | 700 | Thm 1 (three parts), Remark | **Fig 2** (exact mechanism) |
| 4 | Exact-model check and synthetic intervention (`04_synthetic.tex`) | 2.00 | 850 | 0 | **Fig 3** (Exp 2) |
| 5 | Real spectra with assigned neighbourhoods (`05_real.tex`) | 2.00 | 850 | 0 | **Fig 4** (Exp 3), **Table 1** |
| 6 | Limitations and conclusion (`06_limitations.tex`) | 0.75 | 450 | 0 | — |
| — | AI-use + reproducibility statements (outside the count) | — | 200 | — | — |
| **Σ** | | **9.00** | **≈ 4,340** | | 4 figs, 1 table |

Hard rules unchanged from v2: every number traceable to a file under
`results/`; no `\TODO`, no `??`; forbidden list §6; cut prose, never
margins. A reader must understand the claim after Fig 1.

## 2. Section specs

### Abstract (≤ 160 w)
Problem (serial spectral–spatial pipelines: a small per-pixel spectral
encoder feeds a wide spatial model); the claim; what is proved (update
fraction ≤ 1/(1+Mv₀²) at matched fit, reversal error → Φ(m/2σ), removed by
readout normalization); what is observed (`[E2]`: suppression at matched fit
in a ReLU CNN, its dependence on width and readout rate, its absence when
context is uninformative, its removal by normalization; `[E3]`: the
corresponding effect with real spectra in assigned neighbourhoods); one
scope sentence (curvature diagnostics on a production model do not identify
the mechanism; a matched retraining study finds no resolvable freezing
effect). No "within a factor of six". No "certificate".

### 1. Introduction (750 w) — `01_intro.tex`
- Hook (120 w): IR/QCL histology pipelines compress each pixel's spectrum
  with a small encoder and classify with a wide spatial model; practitioners
  pretrain or freeze the spectral stage (`berisha2019deep`,
  `oleary2026spatial`; companion tokenization study, anonymous). When the
  neighbourhood predicts the label, does joint training starve the encoder,
  and when does this matter?
- The claim + the three levels (100 w).
- Contributions (4 bullets ≤ 40 w): (i) Theorem 1 (thm:serial) with the
  three conclusions; (ii) Exp 2: a nonlinear ReLU CNN reproduces `[E2 verdict
  on P1–P4]` with a control that removes the cause; (iii) Exp 3: real spectra
  in assigned neighbourhoods `[E3 verdict]`; (iv) what does NOT identify the
  mechanism: scalar curvature on a production model (Appendix E.1) and an
  unmatched freezing comparison (E.2).
- Related work (250 w), one sentence delta each: gradient starvation
  (`pezeshki2021`), simplicity bias (`shah2020`), modality competition
  (`wang2020`, `peng2022`, `huang2022`, `du2023unimodal`), lazy training and
  balance invariants (`astra_chizat…`→ existing key, `astra_du2018balance`,
  `astra_saxe2014`, `astra_yun2021`), curvature/width/normalization
  (`karakida2019`, `vanlaarhoven2017l2`) only as the reason curvature is
  not our diagnostic, frozen features in practice (`kirichenko2023dfr`,
  `zhang2022layers`). Our delta: an explicit finite-fit account of selective
  ENCODER learning in a SERIAL system with a reversal failure and a
  normalization control, tested by interventions. Never "first".

### 2. Model, task, failure (380 w + Fig 1) — `02_model.tex`
- The serial model: f_θ per pixel S→K, g_φ of width M reading the encoded
  neighbourhood; margin loss; training distribution with two cues that both
  predict the label: a spectral cue in the pixel's own spectrum (requires
  encoder alignment) and a contextual cue in the neighbourhood (readable by
  the spatial model at initialization).
- Definitions (Astra task 3 drafts the prose): MATCHED FIT = first time the
  training loss (margin) reaches a fixed threshold; SPECTRAL LEARNING = the
  encoder's update along the spectral direction as a fraction of the task
  (theorem: (a(T_m)−a₀)/(m−a₀); experiments: encoder energy on u and an LDA
  probe on the encoder output); RELIANCE = accuracy under context reversal;
  FAILURE = error under reversal at matched fit while a spectral-only reader
  has none.
- Fig 1: diagram (pixel spectrum → encoder → spatial model over the 3×3
  neighbourhood; the two cues; the three test conditions iid / reversed /
  context-random). Vector graphic, ≤ 0.25 pp.

### 3. Main theorem (700 w + Thm + Fig 2) — `03_theorem.tex`
Theorem 1 (thm:serial) = Astra's restatement (REFOCUS_PLAN §7.1), general
initialization a₀ < m, v₀ ≠ 0, zero-sum readouts, unit-rate gradient flow on
log(1+e^{−q}), T_m the first hitting time of q = m, immediate stop if
a₀ ≥ m. Three conclusions, each labelled:
 (a) Suppression: 0 < (a(T_m)−a₀)/(m−a₀) ≤ 1/(1+Mv₀²), M(a(T_m)−a₀) →
     (m−a₀)/v₀²; joint–frozen margin gap 0 ≤ q_J−q_F ≤ C₀/(p₀Mv₀²) for
     0 ≤ t ≤ T_m (sup restricted to the fitting interval — Astra item 3).
 (b) Reversal: fitted margin 2a(T_m)−m; error one for a₀ < m/2 once
     Mv₀² > m/(m−2a₀); correct at every width for a₀ ≥ m/2; for
     (a₀,v₀) ~ N(0,σ²I): lim_{M→∞} E_init Err = Φ(m/2σ) (a LIMIT of an
     expectation — item 2); spectral-only comparator error 0.
 (c) Normalization: readout α_M Σ_j γ_j z₂ with α_M = M^{-1/2} gives the
     M = 1 dynamics for every M (Proposition B.7 of v1 → App. A).
Remark (special initialization a₀ = 0, v₀ = 1): D_curv(0) = M exactly,
Ψ(m)/(M+1+2m²) ≤ T_m ≤ Ψ(m)/(M+1), a(T_m) ≤ m/(M+1) — stated as the
special case, never as the general disparity (item 1: general is Mv₀²).
Proof sketch (120 w): balance invariant b² = M(v²−v₀²); Q_M monotone;
convexity for the gap; sign of 2a(T_m)−m.
Interpretation (150 w): replication is an effective learning rate; the
UPDATE is suppressed, not the coefficient (a₀ can be informative); the
normalization control isolates the cause; this is the mechanism the
experiments test, not a statement about production networks.
Fig 2 (`figures/fig2_mechanism.pdf`; from `results/toy_serial_ce_isotropic*.csv`):
(a) update fraction vs M for several (a₀,v₀) with the bound 1/(1+Mv₀²) and
the normalized control (flat); (b) sup_{0≤t≤T_m}(q_J−q_F) with its bound
(label the horizon); (c) expected reversal error vs M with the limit
Φ(m/2σ) (Gaussian sweep: 0.8207/0.8380/0.8397 at M = 64/1024/16384 vs
0.8413 for m = 2, σ = 1). Caption states m and the initialization for each
panel. Reuse `figures_src/make_fig1_serial.py` as the starting point.
DROPPED from v2: Theorem 2's displacement bound and the "within 4.5–6.3×"
sentence (items 11–12 become moot).

### 4. Exact-model check and synthetic intervention (850 w + Fig 3) — `04_synthetic.tex`
- Exp 1 (100 w): 525 non-trivial initializations × widths 1–4096 × m ∈
  {2, 2.9, 4}: closed form vs full (2+M)-parameter ODE, max relative
  discrepancy 1.26e-11 over the four endpoint quantities, zero bound
  violations (`results/toy_serial_ce_isotropic_REPORT.md`). One sentence:
  this is a check of the algebra, not evidence for the claim.
- Exp 2 (750 w): design in REFOCUS_PLAN §4 (data, readiness asymmetry
  0.91 vs 0.63 through a random encoder with equal oracles 0.95/0.95; ReLU
  CNN; full-batch GD, one global rate; matched fit L* = 0.30; arms sp / mup
  / ctxfree / lrmult / frozen; measurements). Results `[E2]`: report P1–P5
  as pre-registered at L* = 0.30, then the labelled secondary analysis at
  0.15 and 0.10 (REFOCUS_PLAN 4.4a), with per-seed Spearman values and the
  trajectories' answer to "transient or persistent". State the amendment
  (widths 2, 4 and fractional multipliers added after a one-seed pilot
  showed saturation at M ≥ 8) in one sentence. Numbers only from
  `results/exp2_summary.csv` / `results/exp2/PREDICTIONS.md`.
- Fig 3 (`figures/fig3_exp2.pdf`): (a) encoder energy on u at L* vs M
  (sp, mup, ctxfree, frozen; seed points + means); (b) accuracy under
  reversal at L* vs M; (c) readout-multiplier panel at M = 32; (d)
  trajectories a_u(t) for three widths (sp) with the loss thresholds marked.

### 5. Real spectra with assigned neighbourhoods (850 w + Fig 4 + Table 1) — `05_real.tex`
Design in REFOCUS_PLAN §5. Say what it is: "a controlled contextual-shift
benchmark built from real spectra"; it does not establish that the
mechanism explains natural clinical shifts. Results `[E3]`: matched-fit
spectral probe and reversal accuracy vs width and readout rate, informative
vs uninformative training context, held-out patients. Table 1: arms ×
(spectral probe, acc iid, acc reversed, acc context-random) at matched fit,
mean ± sd over seeds. Fig 4: the intervention panels as in Fig 3 for the
real-spectra model. If E3 does not support the prediction, the section
reports that and §6 narrows the empirical claim (do NOT drop the section).

### 6. Limitations and conclusion (450 w) — `06_limitations.tex`
- Conclusion first (Astra's sentence, only if E2/E3 support it): "We prove
  this mechanism in a linear spectral–spatial model and find corresponding
  suppression and context-shift failures in the tested nonlinear
  architectures" — name the architectures and conditions.
- Limitations: linear encoder in the theorem; constructed context cues;
  matched-fit stopping is the theorem's convention, not a training recipe;
  one head family per experiment; full-batch GD / plain SGD only; K/S sets
  the initial readiness asymmetry.
- Scope paragraph 1 (production diagnostics, 90 w): on the production
  BlockViT model the initialization curvature ratio is inverted by
  effectively rank-one inputs and its width trend is a normalization-gain
  effect; scalar curvature therefore does not identify the mechanism, which
  is why this paper measures learning outcomes under interventions
  (Appendix E.1; numbers there only).
- Scope paragraph 2 (matched freezing study, 90 w): at width 48 and three
  seeds, an earlier apparent freezing advantage was protocol-confounded and
  neither the contrast nor its change across protocols is resolved; no
  mitigation is recommended (Appendix E.2; Astra's Q6 wording verbatim).
- Open problem (40 w): the mechanism in trained-from-scratch production
  pipelines with natural context statistics.

### Statements: reuse `sections_iclr/statement_ai.tex`, `statement_repro.tex`
(copy into `sections_iclr_v2/`, update the appendix letters).

## 3. Figures (exactly four) and Table 1
Fig 1 model diagram (new, `figures_src/make_fig1_model.py` or TikZ); Fig 2
mechanism (adapt `make_fig1_serial.py` → `make_fig2_mechanism.py`); Fig 3
Exp 2 (`make_fig3_exp2.py` from `results/exp2_summary.csv` + traj files);
Fig 4 Exp 3 (`make_fig4_exp3.py`). Table 1 from `results/exp3_summary.csv`.
Widths ≤ 0.92\linewidth; captions ≤ 80 w; every panel labels model,
optimizer, seeds, threshold.

## 4. Appendix map (`sections_iclr_v2/appendix.tex`)
A. Proofs: Theorem 1 (a)(b)(c) + Remark — verbatim from v1 `appendix_math.tex`
   B.5–B.7 with Astra's restatement; the balance-invariant lemma; nothing else
   from v1 Appendix B (Theorems 1–3, N, P4 lemmas, phase results stay at the tag).
B. Exact-model verification: the isotropic-sweep tables (closed form vs ODE,
   bound slacks, Gaussian reversal sweep), special-init table.
C. Exp 2 details: generator equations, calibration (`calibration.json`
   values), readiness table, arms, all thresholds, per-seed tables, P1–P5
   evaluation, trajectories, the pilot amendment, limitations.
D. Exp 3 details: data, patch construction, patient split, model, arms,
   tables.
E. Scope limitations from v1 (curated, ≤ 2 pp): E.1 production curvature
   diagnostics (from v1 §7.1 + App. E: head hypothesis unmet structurally,
   D_curv 0.46 ± 0.21 at M = 192, Σ_X effective rank 1.07/1.02, BN gain
   trend +0.61 vs +0.006, function-preserving c⁻² scaling; QCL, not FTIR —
   Astra item 13); E.2 matched freezing study (v1 Table 1 + verdicts +
   disclosures; Q6 wording).
F. Verification record (audit scripts actually used by v2 statements only —
   Astra's Appendix F remark).

## 5. Claims ledger (fill as sections are written)

| Claim (section) | Artifact |
|---|---|
| Thm 1(a)(b)(c), Remark | `appendix_math.tex` B.5–B.7 (v1), `code/audits/astra_math_02_checks.py`, `toy_serial_ce_isotropic.py` |
| Exp 1 discrepancy 1.26e-11, zero violations | `results/toy_serial_ce_isotropic_REPORT.md`, `_isotropic.csv` |
| Reversal limit values 0.8207/0.8380/0.8397 vs 0.8413 | `results/toy_serial_ce_isotropic_gaussian.csv` |
| Exp 2 calibration/readiness | `results/exp2/calibration.json` |
| Exp 2 P1–P5, secondary thresholds | `results/exp2_summary.csv`, `results/exp2/PREDICTIONS.md`, `results/exp2/traj_*.csv` |
| Exp 3 | `results/exp3_*` (to be created) |
| E.1 numbers | `results/exp1_8_real_dcurv*.csv`, `exp1_8b_*`, `exp1_8d_*` |
| E.2 numbers | `results/e3c_analysis_runs.csv` |

## 6. Forbidden (grep before submitting)
All of v2 §6, plus: "within a factor of six", "certificate", "diagnostic
framework", "starvation" as a statement about the production model, "we
prove that joint training fails" (unqualified), "universal", "in general
networks", any claim about Exp 3 before its artifacts exist, "FTIR" for the
production data (it is QCL).
