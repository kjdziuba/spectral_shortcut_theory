# Progress Log

Running session-by-session log. Newest entries at the top.

---

## 2026-08-27 — E3c stall diagnosed and fixed: 75x speedup, matrix relaunched

Lanes ran 21h at ~3,600 s/epoch (projected ~120): 27min USER vs 21.7h
KERNEL time per trainer, GPU 0% — per-step GB-scale numpy alloc/free
churn (augmentation copies on 412MB cores) + per-batch page-locking
(pin_memory) inside a 47GB heap. Constant from epoch 1; absent in the
8-core smoke (small heap). Fix: augment=False + pin_memory=False
(identical across arms -> comparisons unaffected; augmentation not
load-bearing for the theory tests). Post-fix: 45-52 s/epoch at h48.
Ops notes: pkill pattern matched own shell twice (use pgrep -f
'pat[t]ern' bracket trick); author's bhargava_rf.py RF baselines (x3)
share the box CPU. 20-epoch pilot curves archived in e3c_pilot_slow/
(already showing the right shape: joint h192 valF1 0.78 theta moved
27%; frozen_pca h48 valF1 0.73 theta pinned, r_rms 0.01-0.09 —
phi alone crushes the residual). Full matrix ETA overnight.

---

## 2026-08-26 (evening) — E3e complete: the normalization, not the preprocessing, is the lever

Four at-init configs (36 measurements each), breast fold 0:

| config                        | lam_th(192) | D_curv(48/192/384) |
|-------------------------------|-------------|--------------------|
| BN + uncentered (E3a baseline)| 471         | 0.33 / 0.46 / 0.70 |
| BN + centered                 | 448         | 0.34 / 0.47 / 0.74 |
| no-BN + uncentered            | 194         | 0.58 / 1.18 / 1.31 |
| no-BN + CENTERED              | 194         | 0.58 / 1.18 / 1.31 |

Three findings:
1. **Input centering is a NO-OP for this architecture class** — with
   the internal wn_norm present it re-standardizes the input; without
   it, the post-projection BatchNorm2d cancels constant input shifts
   exactly (linear proj + BN: a global input shift becomes a constant
   feature shift, which BN's centering removes; its Jacobian
   annihilates the constant direction). The no-BN centered run is
   IDENTICAL to uncentered to 4 digits despite sigma_lam1 dropping
   0.93 -> 0.042 (22x). My "centering restores disparity" prediction
   was WRONG as stated — 04's sentence revised same session to the
   correct statement (the lever is the normalization choice; internal
   normalizations define the effective regressor).
2. **The per-wavenumber BatchNorm (wn_norm) INFLATES the spectral cap
   ~2.4x** (lam_th 471 -> 194 without it): dividing mean-dominated
   wavenumbers by their tiny std amplifies the shared-shape component
   (post-BN lam_max(Sigma) ~ 1279 vs raw 0.93).
3. **Without wn_norm, D_curv crosses parity at production width**
   (1.18 at M=192, 1.31 at 384): the E3a "inversion" is partly an
   artifact of the production model's own normalization choice. The
   E3c joint_linear arm (spectral_norm=False per Assumption-1
   hygiene) is EXACTLY the no-BN config -> its at-init D_curv ~ 1.18,
   coherent for Section 8.

Honest scope: all at-init; the "effective regressor after internal
normalization" framing needs one careful paragraph in P2, not a
theorem change (the cap lemma is about the actual f_theta input and
remains true; the REALIZED top depends on downstream Jacobians, which
is what rem:irrelevance/Section-8 prose must carry).

E3c lanes launched (lane48: 9 adamw + 2 sgd; lane192: 9) after
runner smokes passed (frozen theta_disp=0 exact, counterfactual EGR
~0.9, joint theta moving). frozen_pca is lane48's first run
(deliberately: only never-smoked arm). Builder agent terminated after
delivering post-commit mkdir hardening (audited, committed).

---

## 2026-08-26 (later) — Review round 2b (correct file): converged, high-value; Tier-1 fixes applied

All three reviewers passed the sentinel gate — the round is VALID.
Gemini's output is DEGENERATE (boilerplate "Standard processing
applied" x6, one finding attacking a claim the packet no longer
makes, outlier 75/85% numbers) — discarded; optional fresh-chat redo,
low priority. ChatGPT (weak reject, 33% now / ~60% ceiling) and
Fable (weak reject, 15% / 40%) CONVERGED: math now sound; ceiling set
by experiments + claims that outrun theorems. 9/10 fixes verified
genuinely resolved by both.

**Headline catch (ChatGPT #1): weight decay breaks the
Theorem-2-matching arm.** With L2 decay on theta the flow gains
-2*alpha*theta uncontrolled by the envelope (r=0 counterexample).
rem:containment's "applies to the modified flow" was FALSE; E3c's
AdamW wd=0.05 default would have invalidated the displacement test.
FIXED in text; E3c: theta in zero-wd param group in all
theorem-matching arms.

**Tier-1 fixes applied this session** (commit this session):
A.3 TODO stubs deleted (honest pointer paragraph); weight-decay
correction + theta-no-wd protocol in rem:containment; max-margin
log(T) claim restricted to homogeneous nets; Clarke-selection
clause added to the a.e. formulation (every limiting Jacobian
inherits the cap); "entirely free to travel" replaced by the honest
constants-asymmetry statement (both reviewers: the phi-bound is
SMALLER at measured widths); zero-constant case splits (C_theta=0,
C_G=0); rem:not_pca restricted to training time (at init the GGN is
label-free input-Gram arithmetic — Fable F2); 04 feature-
normalization paragraph rewritten (input transformations MOVE the
cap; centering predicted to restore disparity — ties to the 19-32x
measurement); adaptive-optimizer paragraph corrected (preconditioned
spectra; theorems scoped to GD/flow); MY ERROR fixed (degenerate
q_h=Theta(1/N) claim wrong for N >> M_h — floor is Theta(1), no
width scaling; Fable F8); squared-product-bound wording; tr(Sigma_X)
constants clause.

**Tier-2 (UNFREEZE DECISION, awaiting author): two one-paragraph
additions both reviewers converged on:**
(a) Loss-attribution corollary: theta's share of the training-loss
decrease <= B_T^2 C^2/(mu * DeltaL) — elementary, both suggested.
(b) Fable's class-prior residual-rate proposition: the witness
direction is the class-prior residual component; via r-dot =
-H_tau(JJ^T)r its initial fitting rate is Omega(M) — the first
cheap PROVED link Thm1 -> envelope for one residual component.
Sanity-checked; would need in-house adversarial verification, then
freeze v1.1.
Queued P3/P4 (from round): Assumption-7 init/trajectory split;
06-intro/EGR alignment; Remark-39 stale candidate-cause update;
Prop-34 R-conditional + c2>=c1*M0 + Phi_reg-a.s. clauses; 02 fixes
(Huang theorem-level; O'Leary mischaracterization); "freezing
prescription" split (Thm 2 predicts joint~frozen-random; pretrain-
and-freeze is a separate empirical escape route — ChatGPT #3).
E3c additions: log ||dg_phi/dZ||_op along training (containment as
tested hypothesis, Fable F6); theta-no-wd groups.

---

## 2026-08-26 — Review round 2: stale-attachment mix-up diagnosed; standing findings triaged; centering measurement

**Mix-up**: ChatGPT + Fable both audited the OLD theory_scope_2026-08-20.pdf
(ChatGPT's header names it; Fable's F1 quotes old text verbatim and itself
diagnosed the mismatch). Cause: continued chats still held the stale
attachment. Verified the 2026-08-25 packet DOES contain all ten fixes
(sentinel greps). Their theory re-audit findings are therefore VOID where
they re-report fixed items — which is silver lining: ChatGPT independently
re-derived the same defects a second time (confirms fix targets were real)
and re-validated Prop 35 / witness / Kronecker / Thm-2 integration as sound.

**Standing findings (prompt-based, valid regardless of PDF), triaged:**
- F2 (Fable, blocker-class, GENERATIVE): the rank-1 Sigma_X + 0.954
  alignment are properties of UNCENTERED nonnegative spectra; at init the
  linear module's GGN is label-free input-Gram arithmetic (Wu et al.
  "Dissecting Hessian" arXiv:2010.04261 anticipate u (x) E[x] structure;
  Papyan 2019/2020). MEASURED same day: mean-centering on valid pixels
  deflates lam_max(Sigma_X) 19-32x per core (centered eff rank ~1.7-2.0)
  -> predicted D_curv rises ~20-30x under centered preprocessing, i.e.
  ~10 at M=192: the inversion likely REVERSES under a preprocessing
  choice. Response: (a) report input preprocessing explicitly; (b) add
  mean-centered at-init exp1_8 variant + input-energy control ratio next
  to the 19x contrast ratio; (c) reposition C4 as
  confirmation-in-new-domain + quantification, citing Wu/Papyan; (d)
  centering becomes a PREDICTED MITIGATION LEVER (theory-consistent:
  the cap is set by lam_max(Sigma_X)). Note: model's wn_norm BN does NOT
  center tissue pixels (padding dilutes the batch mean) — explains why
  post-BN eff rank stayed ~1.2.
- F3 (Fable, major, real): Thm 2's bound applied to phi is LARGER than
  for theta at measured widths (ratio of bounds = sqrt(D_curv) < 1) —
  "selective laziness" not derivable from the bound alone at these
  constants. Adopt their loss-attribution substitute: fraction of loss
  decrease attributable to theta <= B_T^2 C^2/(mu * DeltaL), computable
  from E3c logs; report honestly even if vacuous.
- F4 (Fable, major, real): Thm-2 bound with B_T ~ sqrt(544) exceeds
  ||theta_0|| — displacement-vs-bound plot would be vacuous. Log ALSO
  feature-space displacement ||Z(t)-Z(0)|| and projection of
  theta(t)-theta_0 onto contrast directions (the direction-wise
  dynamical test). Report bound/realized ratio explicitly.
- F5 (Fable, major, real): Adam vs gradient-flow mismatch — under Adam
  small gradients do not imply small steps. E3c: add SGD control at one
  width (A2-vs-A1 pair) as the theorem's actual regime; fix 04's
  adaptive-methods sentence in P3 (preconditioned dynamics need not
  respect the disparity).
- F10 (Fable, major, sharp): the sublinear slope is DIAGNOSABLE —
  measure lam_max(S_h^p) directly at each width (cheap Gram) to see if
  q_h is effectively width-dependent, and whether the head-vs-full-phi
  measurement mismatch explains the deficit. Added to E3c probes.
- ChatGPT-7 (major, real): mu in the envelope is not canonical (any
  smaller mu also satisfies it). E3c defines the estimator explicitly:
  mu-hat := sup{mu : ||r(t)|| <= C-hat/(1+mu t) for all t <= T} with
  C-hat = ||r(0)||, reported with fit residuals.
- New must-cites (P4): Zhang et al. 2024 "Why Transformers Need Adam"
  (block heterogeneity); Karakida et al. 2021 softmax extension; Wu et
  al. Dissecting Hessian; Papyan 2019/2020. Huang 2022 miscast = A27
  (already tracked); Coil/Cheney = A44 (already tracked).
- Residual defects in NEW packet caught by F8/F9, FIXED this session:
  weight-decay route sentence rewritten (bounded iterates give finite,
  not width-uniform, L~ — width-uniformity is the assumption's modeling
  content); ass:reg gains the piecewise-smooth a.e. clause. Recompiled;
  packet theory_scope_2026-08-26.pdf regenerated (26 pp, sentinels
  verified).

**Redo protocol**: re-send task 1 (theory re-audit) ONLY, in same chats,
with theory_scope_2026-08-26.pdf and a sentinel-check instruction; the
measurement/design findings stand and are folded into E3c. Gemini's
review not yet delivered by author.

---

## 2026-08-24 (later) — E3b + E3d complete: the rank-1 diagnosis is CONFIRMED and quantified

**E3d (prostate QCL, 48 rows, results/exp1_8_real_dcurv_prostate_qcl.csv):**
full replication. Cap slope -0.012 (pred. 0.0); lam_phi slope 0.45;
D_curv 0.44/0.52/0.69/1.09 at M=48/96/192/384 (parity ~M=384, earlier
than breast's ~1300); Sigma_X eff rank 1.02/942 — rank-1 input is
MODALITY-WIDE. Commit 5da0f3e.

**E3b (breast, 3 seeds x 4 cores, M=192; results/exp1_8b_{spectra,
summary}.csv; machinery code/hessian/lanczos.py validated 1e-16):**
1. **Alignment 0.954 +/- 0.017**: the spectral block's top eigenvector
   IS the mean-spectrum reader (its proj-weight component projects 95%
   onto v_data, the top eigenvector of the post-BN input Gram).
   Restricted-block check: lam over the {u (x) v_data} subspace
   reproduces lam1 (e.g. 359 vs 386) — the top curvature lives there.
2. **Spectrum shape**: lam_k/lam_1 = 1.00/0.54/0.43/0.26/0.07/0.03 at
   k=1/2/3/5/20/40. Not a single spike, not a 64-flat shelf: a fast
   decay. **Top-40 of 61,108 directions hold 83% +/- 11% of the whole
   block's trace** (0.065% of the space holds ~5/6 of the curvature).
3. **Crossover rank 2-5**: lam1(phi)=174 beats every spectral
   eigenvalue from rank ~2-5 onward, in all 12 units. So the spatial
   block dominates 99.99% of the spectral parameter space; D_curv's
   inversion is caused entirely by theta's top ~3 mean-reader
   directions.
4. **Contrast starvation (THE number)**: max curvature along the
   CancerEpi-vs-CAS class-contrast direction (orthogonalized to
   v_data) = 28 +/- 9, i.e. **12-28x (mean ~19x) below the mean-reader
   direction** and ~6x below the spatial block's top. Measured by exact
   64x64 restricted GGN blocks (64 matvecs/direction). LIMITATION: only
   the CancerEpi-CAS pair was measurable — the densest fold-0 train
   cores contain no NormalStroma/NormalEpi pixels (class counts logged
   per row); a follow-up with stroma-containing cores would add the
   headline CancerEpi-NormalStroma pair.
5. Seed effect: lam1_theta spans ~190-830 by seed (the random spatial
   init's L~ moves the whole theta block, per Lemma opcap); alignment
   and decay shape are seed-stable.

**Refined Section 8 story (P2), one sentence:** the spectral module's
apparent curvature wealth is an artifact of ~3 directions that read the
non-discriminative mean spectrum; outside them the spatial block
dominates everything, and the direction that actually separates cancer
from CAS has ~19x less curvature than the mean-reader — direction-wise
starvation, invisible to top-vs-top D_curv, measured directly.

Theory remains FROZEN and consistent: cap confirmed twice more
(slopes -0.002/-0.012), D_curv >= c1*M satisfied everywhere (c1 tiny
via lam_max(Sigma_X), now understood mechanistically).

---

## 2026-08-20/24 — Phase E opens: GGN instrument built, Exp 1.8 run on real data — INVERTED D_curv finding

**Instrument (code/hessian/ggn.py, NEW).** True Gauss-Newton block
power iteration: Gv = J^T[H_tau(Jv)]/N via JVP (double-backward trick)
+ closed-form softmax Jacobian + VJP. Validated to 1e-12 against a
brute-force materialized J^T H_tau J on a toy model; PSD-checked;
gradient identity grad L = J^T r matches autograd to 2e-17. Closes
A24: code/hessian/eigenvalues.py measures LOSS-HESSIAN blocks, which
differ from G by the uncontrolled functional-Hessian term — every
prior eigenvalue number measured the wrong matrix. Gotcha: PyTorch's
fused SDPA kernel has no second derivative
(_scaled_dot_product_efficient_attention_backward) — GGN on any
transformer requires forcing the math backend (math_attention() ctx).
EGRLogger gained from_param_groups() (BlockViTv2 has no
.spectral/.spatial) with a frozen-arm guard against vacuous EGR=0.

**Checkpoint reality (Explore agent, verified).** NO jointly-trained
BlockViT v2 checkpoint exists anywhere local — joint runs lived on
CSF3, only stdout came back. Sole survivor: frozen breast fold0
(K=128) in deep_dream/results/train3d/blockvit_v2_mlp_breast_fold0/
(needs sibling spectral_reduce.pt + strict=False). experiments_final/
v2.0/ is EMPTY (v1.0 holds the 1-D spectral models only). So Exp 1.7
(EGR on trained variants) requires RETRAINING; full matrix = 30 runs
~ 10 GPU-days = does not fit before Sep 18. PILOT proposed (breast
fold0, joint vs frozen, ~50 epochs, ~1 day): EGR collapse is an
early-training phenomenon, no need for convergence. AWAITING GO.

**Exp 1.8 (code/experiments/exp1_8.py, NEW) — needs no checkpoints:**
Theorem 1/def:dcurv are at-INIT statements. Production BlockViTv2 on
real breast cores (fold0 train, 4 densest cores 9-26% labelled),
width sweep M in {48,96,192,384} x 3 seeds x 4 cores = 48
measurements, ~25 power iters each, <7 GB. CoreDataset eager-loads
its whole split (~0.4 GB/core) — runner writes a trimmed split file
(densest-core selection, core_oversample=6).

**Results (results/exp1_8_real_dcurv.csv):**
- lam_theta FLAT in width: log-log slope -0.002 (predicted 0.0).
  Lemma opcap CONFIRMED on real data — the proved half validates.
  Per-seed lam_theta in {~200, ~390, ~650} (seed-dependent L~ of the
  random spatial net, exactly as the lemma's constant allows), flat
  across M within every seed.
- lam_phi grows sublinearly: slope +0.38 (predicted 1.0) — same
  undershoot family as synthetic Exp 1.1 (0.70).
- **D_curv < 1 at every buildable width**: 0.33/0.40/0.46/0.70 at
  M=48/96/192/384. At production M=192: D_curv ~ 0.46 — the
  spectral block has MORE top-curvature than the spatial block.
  Extrapolated crossover D_curv=1 at M ~ 1,300 (~7x production).
  Section 8's old "predicts large disparity" headline is DEAD.
- Confounds ruled out: (a) BatchNorms inside spectral_reduce carry
  lam ~ 0.16/0.02 vs the proj conv's ~400 — the big curvature IS the
  theory's W^T x map; (b) padding/label-density: corr(valid_px,
  D_curv) = +0.19/-0.02/+0.13/+0.28 by width — weak, inconsistent,
  not the driver.

**Diagnosis: Sigma_X is effectively RANK ONE.** On real tissue,
lam_max(Sigma_X)/trace = 0.937, effective rank 1.07 of 942 dims
(raw), 1.20 after wn_norm (lam_max ~ 1.28e3). Every tissue spectrum
shares ~the same shape; discriminative chemistry is a tiny
perturbation. Consequence inside the theory: the cap C_theta =
L~^2 lam_max(Sigma_X) is HUGE, so c_1 = c_phi/C_theta is tiny and
the (true) bound D_curv >= c_1 M is uninformative at buildable
widths. Theorems NOT falsified — cap confirmed, lower bound
satisfied — but the top-vs-top ratio cannot carry the real-data
story: theta's top direction is the (non-discriminative) mean-
spectrum reader; the ~941 chemistry directions are near-flat. The
starvation is DIRECTION-WISE, invisible to D_curv by construction
(cf. rem:irrelevance_vs_starvation — this is its concrete real-data
instance). TO CONFIRM: measure G_thetatheta's top-k spectrum +
alignment of its top eigenvector with the mean-spectrum direction
(deflation power iteration; ~30 min — proposed next).

**Paper impact (P2 scope, theory stays FROZEN):** Section 8 rebuild
tells: (i) cap validated on real data (slope -0.002 — a genuine
win for the proved half); (ii) lam_phi growth sublinear, reported
honestly; (iii) inverted top-ratio + rank-1 Sigma_X as a FINDING
about spectroscopy (one direction hogs the spectral module's
curvature budget; the chemistry lives in the flat directions);
(iv) real-data dynamical evidence shifts to EGR (Exp 1.7 pilot).
Feeds D2's asymmetry story: "the cap is set by the data" is now
concrete — the data sets it VERY high via its rank-1 structure.

---

## 2026-08-20 — T5 external half: three-LLM review returned, fix round applied, THEORY FROZEN

Three fresh-chat reviews (identical prompt + 25pp packet): ChatGPT deep
research, Claude Fable, Gemini. Verdicts: 2x weak reject (+1 truncated,
same tenor); self-estimated ICLR odds 22-35% on theory alone. The
headline: NO reviewer found a mathematical error in any displayed
proof. All three independently re-verified Prop 35 Steps A-E, the
witness bound, the Kronecker identity, and Thm 2's integral. The
protocol also worked as designed: agreement was informative, and the
round produced its customary confident misreading (Claude — which
admitted it could not open the PDF — invented a "hidden M_0 width
threshold" that is literally in the proposition statement, and
misattributed "modality laziness" to Wang 2020; workflow-verified: Du
et al. ICML 2023 coined it).

**Unanimous finding (all 3): no proved geometry->dynamics bridge; Thm 2
integrates an assumed envelope; mu(Cg) growth is hypothesized.** This
is the paper's own stated position (05 says so in four places), so it
consumed no fix — it is a POSITIONING problem: the contribution must be
sold as (cap + envelope + verified instance + open bridge), with the
bridge as explicit open problem and the E-phase mu-vs-capacity
measurement (D2/E4) as its empirical test. Sharpened 05's "genuinely
enters the dynamics" to the precise cap-only statement.

**Confirmed real defects, all fixed (commit this session):**
1. ass:subg claimed lambda_max(Sigma_X) -> lambda_max(Sigma) with no
   independence across pixels (ChatGPT's duplicated-pixel
   counterexample is valid). Restated as the empirical bound actually
   consumed (lambda_max(Sigma_X) <= Lambda_X); population story moved
   to rem:subg_population, marked not-used-in-proofs.
2. Bai-Yin misuse (ChatGPT + Gemini, same counterexample): a fixed-K
   expansion matrix has op norm Theta(sqrt(M/K)), NOT Theta(1). Remark
   rewritten: proportional-dimension edge (YBK 1988 + Bai-Yin 1993,
   both now cited correctly) vs composition-cancellation (proved in
   expectation in Prop 35 Step D); ass:inputlip imposed, not derived.
3. Karakida odd-activation footnote (ChatGPT): verified verbatim
   against PMLR PDF — they explicitly scope OUT the kappa_2=0 case
   ("outside the scope of this study"). Our "mirroring their surviving
   kappa_1/T term" attribution deleted; footnote now quoted verbatim,
   our Theta(1/N) degenerate behavior derived independently.
4. Gemini's "blocker" (ReLU vs C2 ass:reg): hypothesis-hygiene, not a
   proof error — GGN needs only first-order Jacobians (a.s. defined at
   Gaussian init). Thm 2 restated for absolutely continuous
   trajectories satisfying the flow a.e. (also fixes ChatGPT's
   missing-regularity-hypothesis minor); new rem:relu_scope.
5. Expectation placement in Thm 1 proof made explicit (E[ratio] >=
   E[numerator]/deterministic cap) in 04 + supplement combine.
6. D_curv := +infty convention when the spectral block vanishes.
7. lem:interlacing deleted (orphan; flagged by 2/3 reviewers + tracked
   A38). 8. 05/supplement TODO promises ("discussed in the
   supplement") replaced by honest rem:thm2_open. 9. LayerNorm
   compactness claim softened to within-Phi_reg boundedness.

**Best substantive point (ChatGPT), turned into theory-strengthening
honesty:** Prop 35's spike comes from the bias-induced mean feature
direction — set all z_n = 0 and the width-linear curvature persists
with zero-information features. So D_curv measures READINESS TO MOVE,
not usefulness. NEW rem:instance_interpretation states this candidly
and flips it: the capacity advantage predates the data — that is
exactly why it is a shortcut hazard. One-sentence echo in 04's
interpretation remark.

**Verification workflow** (wf_07653a70-85f, 7 agents, all confirmed
against primary sources): Karakida footnote verbatim; Lim/Kim/Moon
NeurIPS 2025 Spotlight is REAL (must-cite, added to P4 with delta);
Zhang/Bengio/Singer JMLR 2022 real ("robust"/"critical"); laziness =
Du et al. ICML 2023; Coil & Cheney = AutoML 2025 PMLR 293,
hypothesize-not-show; Pezeshki Thm 2 characterization accurate
(NTK-linearized + coupling + s1^2>s2^2); largest-eigenvalue edge =
YBK 1988 (bai1993limit is the smallest-eigenvalue paper) — bib fixed,
yin1988largest added.

Compile clean (39pp; only known fig:exp11_paired undefined — P5).
**THEORY STAMPED FROZEN.** Next: Phase E (launch Exp 1.7/1.8 first),
with remaining reviewer points (novelty delta, related work) scoped to
Phase P.

---

## 2026-08-19 (later) — T5 freeze gate: in-house half complete

**Round 1** (wf_406b7519-2b2; 5 hostile lenses, citations lens + all 9
workflow verifiers died on the session limit — every finding manually
verified on Fable instead; an index-shift bug also mislabeled lenses in
the output, corrected during triage):

The two substantive blockers, both answered with NEW theory per the
author's adjust-experiments-to-theory directive:
1. "Theorem 1 is arithmetic on two assumptions — no proved instance."
   -> NEW prop:verified_instance: two-layer ReLU head at Kaiming init,
   sigma_b > 0. ReLU's positive mean (E ReLU(N(0,s^2)) = s/sqrt(2pi))
   gives every feature vector a shared direction of Theta(M) energy —
   the head hypothesis PROVED via mean-feature concentration.
2. "ass:residual is circular." -> NEW rem:not_circular: the assumption
   constrains the RESIDUAL, not theta; the bound caps theta's travel on
   ANY trajectory consistent with the loss curve, including ones where
   theta does part of the fitting. Contrapositive: substantial spectral
   learning by the fitting time REQUIRES slow fitting. Independently
   checkable from a run's own residual curve.
Also: bounded-C-along-the-family condition added to the capacity
asymptotics (ass:residual refuses uniformity, so the asymptotic reading
cannot be free); "Relation to lazy training" + chizat2019lazy (selective
module-wise laziness vs global linearization); 06 opening rewritten
counterfactually (A41 killed); D_curv top-vs-rest spectrum honesty;
rem:param_scope (standard-parameterization statement, no muP transfer);
tau-sensitivity dangling promise deleted; ~20 smaller repairs
(2*alpha*I, kappa(Sigma), B_T forward-refs, N_cls typesetting, ...).

**Round 2** (wf_38ae8bc0-c40): proposition verified — Steps A/B/D and
the witness bound all CONFIRMED sound; 2 majors in MY probability
bookkeeping caught and repaired:
- Step C conditioned on F_1 while bounding an F_1-measurable quantity
  by its unconditional mean — fixed with a separate Markov event.
- Part (ii)'s per-input Jacobian cap cannot deliver ass:inputlip's sup
  over inputs (sqrt(log N) by the union route). UPGRADED instead of
  patched: trace-level cap E[lambda_max(G_tt)] <= N_cls sigma_w^4
  tr(Sigma_X), free of M and N, plus NEW part (iii): the INSTANCE-LEVEL
  DISPARITY LAW, E[D_curv] = Omega(M), proved outright — the
  proposition now delivers the theorem itself for the instance, not
  just its hypotheses.
Citations make-up lens: Karakida Thm 4's output IS fixed C = O(1) (only
the input side excludes our heads) — both cautions corrected; Yang-Hu
softened to what Tensor Programs IV proves; Soudry attributed to linear
predictors with Lyu-Li 2020 added for homogeneous nets.

Commits: 50f26cb, c09676d, 41c74f8. Compiles clean.

**T5 status: in-house gate CLOSED. Full freeze awaits the author's
external-LLM review ritual on Sections 3-6 + supplement (especially
prop:verified_instance — fresh probability written this session), then
one fix round, then FROZEN and Phase E launches (Exp 1.7/1.8 first).**

---

## 2026-08-19 — T4 capacity ratio recounted from the real model

**Every number in the capacity claim was wrong; all three are now
replaced by counts from the instantiated model** (script saved at
`code/count_blockvit_params.py`, runs against
`spectral_tokenization/side_project/models/blockvit_v2.py`):

| config | C_f | C_g | ratio |
|---|---|---|---|
| K=64 (end-to-end baseline) | 61,108 | 18,271,540 | **299** |
| K=128 (matched to frozen variants) | 121,588 | 21,472,564 | **177** |

- **13M vs 18.3M RESOLVED**: total(K=64) = 18,332,648 = the companion's
  18.3M. The companion quotes the TOTAL model size; our draft's "C_g =
  13M" was simply an undercount. No contradiction between the papers.
- **The 325 fossil identified exactly**: 314 x 128 = 40,192 (a
  SINGLE-channel spectral count) divided into the undercounted 13M
  gives 323 ~ 325. The real spectral module takes 3 channels
  (3 x 314 = 942 inputs), so C_f is ~3x larger than the fossil assumed.
  The audit's "108" was also wrong — it used the paper's own two wrong
  numbers. Only 299 is computed from the model.
- **Theta(M^2) confirmed for BlockViT**: transformer 5.34M +
  ConvTranspose 9.44M, both quadratic in M = hidden_dim = 192
  (patch_embed 3.15M is only linear in M and not dominant). So the
  sqrt(C_g/C_f) corollary of Theorem 1 LEGITIMATELY applies here —
  unlike Exp 1.1's SpatialMLP, where C_g = Theta(M) and slope 1.0
  applies. The paper now has one architecture from each class, which
  strengthens the A6 story rather than complicating it.
- Prediction for Exp 1.8: D_curv >= c1' sqrt(ratio) ~ 13-17 c1',
  equivalently c1 * 192 in the theorem's native width variable.
- New Table tab:capacity in Section 8 with the full block breakdown;
  kappa deleted from Section 8 (A5); 03 updated to ~300 and "roughly
  two orders of magnitude"; Exp 1.8 restated in the inventory around
  D_curv; Section 8 provenance corrected to QCL (A52) and the
  architecture attributed to O'Leary et al.

**Next:** T5 THEORY FREEZE GATE — **switch back to Fable first.**

---

## 2026-08-18 (later) — T3 theory-hygiene batch

**All eight T3 items done** (A12, A13, A23, A35, A36, A22, A61; A39 was
already done in T2), plus six opportunistic fixes:

- **A12 normalization**: Sigma_X is now unambiguously the MEAN empirical
  per-pixel Gram (eq:sigma_x), the loss is 1/N-averaged over all N =
  N_data*H*W pixels, and r/J carry the matching N^{-1/2} normalization
  so grad = J^T r holds exactly. Consequence: **C_theta = L~^2 *
  lambda_max(Sigma_X)** — the factor K is GONE. It was pure slack: the
  Kronecker identity (dZ/dtheta)^T(dZ/dtheta) = I_K kron Sigma_X is
  EXACT, and Kronecker replicates the spectrum rather than accumulating
  it. Every constant in the paper is now dataset-size independent.
- **A13 Jacobians**: 03 now defines both objects once — logit Jacobian J
  and residual Jacobian J^res = H_tau J — fixes the GGN convention
  G = J^T H_tau J (eq:ggn), and states the transfers (H_tau^2 <= H_tau
  <= I) so lem:opcap covers all three conventions. eq:chain remark
  corrected re where the softmax enters.
- **A23 Thm 2**: B -> B_T (finite horizon) with explicit trajectory
  containment in the hypotheses, plus rem:containment acknowledging
  that in the unregularized separable regime ||phi|| diverges, naming
  the two settings that restore the cap (weight decay / muP), and
  noting the max-margin limit costs only a log factor
  (eps log^2(1/delta)).
- **A22 Prop 6.1**: DOWNGRADED from a proposition to an unnumbered
  heuristic (eq:egr_init) with all three failure modes stated
  (E[ratio] vs ratio of E, muP-dependence, bottleneck ignored) and the
  in-manuscript TODO removed. Its only load-bearing role — setting the
  scale for the RELATIVE alert threshold — survives, since the alert
  compares against each run's own measured EGR(0).
- **A35/A36/A61**: D_curv evaluated at random init and tied to
  eq:block_hessian; assumption lists made explicit everywhere (ass:gap
  now flagged as not used in any proof, only delimiting the regime of
  interest); C and mu declared trajectory/architecture-dependent.
- **Bonus**: A57 (lem:schur -> lem:opcap project-wide rename — the
  fossil that caused Section 2's misdescription), A54, A14 third site,
  A16 (slow-fast manifold scrubbed), A37-partial (Section 4 retitled
  to "Capacity-Induced Module Curvature Disparity"), and author
  directive D4 implemented as rem:loss_scope (how much of each theorem
  depends on cross-entropy: Thm 1 essentially loss-agnostic, Thm 2's
  polynomial envelope CE-on-separable-specific).

**Verification round (wf_405ae374-c50) found 4 majors — all fixed in
commit 452ad55, and one led to a strictly better proof:**

1. **Numerator convention break.** lem:phi_scaling was stated for the
   LOGIT Gram J^T J while D_curv is defined on the GGN block
   G = J^T H_tau J. Since G <= J^T J, a lower bound on the logit Gram
   does NOT lower-bound the numerator — the displayed "=" was false.
   Lemma, Theorem 1 proof, and combine step now all use G_phiphi;
   logit and residual Grams follow as corollaries.
2. **The 1/N in the witness constant** contradicted the new
   dataset-size-independence claim. FIXED BY A BETTER WITNESS: instead
   of one pixel's feature vector, use u = top eigenvector of the mean
   feature Gram S_h = (1/N) sum_n h_n h_n^T. Then
   lambda_max(G_phiphi) >= c_p * lambda_max(S_h) with ALL N terms
   contributing — no 1/N anywhere.
   **The head hypothesis becomes lambda_max(S_h) >= q_h M_h, which is
   EXACTLY Karakida et al.'s kappa_2 > 0 condition** (kappa_2 is built
   from cross-input activation correlations q_st). Their footnote 1
   excludes odd activations with sigma_b = 0, where q_st = 0 and
   kappa_2 = 0 — our hypothesis degenerates on precisely the same
   family. Independent consistency check on both.
3. **Symbol collision**: supplement used J_phi for the residual
   Jacobian against 03's naming. Rewritten as an explicit
   three-convention corollary.
4. **prop:egr_monotonic** dropped ass:subg + containment and claimed a
   uniform-in-t bound that rem:containment explicitly warns against.
   Restated for t <= T with B_T.
Plus minors: N^{-1/2} bookkeeping for dZ/dtheta declared in 03 and used
in supplement Step 1; eq:chain brace label corrected; J_Z
block-diagonal-over-images step stated; B = sqrt(C_theta) in 06.

**Re-verification (wf_81843e59-42d) found 4 more majors — fixed in
commit 1d62431 by restructuring, giving the FINAL form of the proof:**

- The mean-Gram witness took `c_p := min_n p_min^(n)`, which RELOCATED
  the N-dependence into the constant instead of removing it (a union
  bound over N pixels forces c_p ~ exp(-C sqrt(log N))) — while the
  text claimed "no factor of N appears". Also c_p was overloaded with
  three incompatible meanings and the expectation step swapped them.
- **FIX — weight, don't minimize.** Define the SOFTMAX-WEIGHTED
  feature Gram S_h^p := (1/N) sum_n p_min^(n) h_n h_n^T. Then
  **lambda_max(G_phiphi) >= lambda_max(S_h^p) is DETERMINISTIC and
  hypothesis-free** — one identity plus one application of Step 2,
  valid at every parameter value. No minimum, no union bound, no N.
  All probabilistic content now sits in ONE clearly stated hypothesis:
  lambda_max(S_h^p) >= q_h M_h whp at init.
- The head hypothesis had been advertised in three places as a
  per-vector NORM condition (||h||^2 >= q_h M_h), which is strictly
  WEAKER than the lambda_max condition consumed — weaker by a factor
  of N, since per-vector norms bound only tr(S_h). Cross-input
  CORRELATION, not per-vector norm, is what makes the top eigenvalue
  grow; the supplement now says so explicitly.
- Main-text remark still described the retired single-pixel witness.
- Karakida correspondence softened: on the degenerate family both
  bounds lose their sample-size-independent leading constant rather
  than vanishing ("exactly the same family" was overstated).
- Kronecker ordering ambiguity removed by writing the witness as the
  rank-one MATRIX V = c u^T with ||V||_F = 1 (convention-free).

Compiles clean; only fig:exp11_paired remains undefined (A10, Phase P5).

**Next:** T4 ratio recount from the real BlockViT v2 checkpoint (A4/A5),
then T5 freeze gate — **switch back to Fable for T5**.

---

## 2026-08-18 — Phase T1+T2 complete: integrity batch + width-native Lemma 4.1 proof

**T1 integrity batch (A2, A3, A30, A44, A56, A58) — DONE:**
- Verified the actual Karakida AISTATS 2019 paper (PMLR 89:1032-1041)
  page-by-page: theorems are plain-numbered 1-7; Thm 1 = mean
  eigenvalue O(1/M), Thm 4 = lambda_max = alpha((T-1)/T k2 + k1/T) M
  = Theta(M); exactly 2 figures; squared loss, linear outputs,
  variance-scaled Gaussian init. All five fabricated citation sites
  (Thm 3.1/4.2/5.1) and the invented "Figures 2-3, exponent [0.5,0.8]"
  attribution removed.
- Bib metadata fixed: karakida (published AISTATS version), soudry
  (JMLR 19(70):1-57), huang2022 (ICML/PMLR 162), peng2022 (full
  authors), oleary (from companion bib: Anal. Chem. 98(4):2743-2755,
  DOI 10.1021/acs.analchem.5c04765), coil2025 (verified: Collin Coil &
  Nick Cheney, PMLR 293), bozzo (DOI). Added bai1993limit +
  vershynin2018high; A58 Bai-Yin cite replaces Marchenko-Pastur for
  the Theta(1) op-norm claim in 03. Unused-entry pruning deferred to P4.

**T2 width-native proof rewrite (A1, A6) — DONE, adversarially verified:**
- Supplement Lemma 4.1 proof fully rewritten. First version routed
  through a "contrast network in Karakida's class" + Thm 4; a 3-agent
  adversarial workflow (wf_4d095e3f-252) REFUTED that route: Thm 4
  requires ALL layer widths proportional to M, and degenerates
  (alpha -> 0) for fixed bottleneck input K + single output. Final
  proof is a half-page ELEMENTARY WITNESS argument: head-weight
  direction v = c kron h(x_1)/||h(x_1)|| (c a unit contrast perp 1)
  gives lambda_max >= p_min ||h||^2 / N = Omega(M) since ||h||^2 =
  Theta(M) at variance-scaled init. Softmax handled exactly via
  H_tau >= p_min P_perp (variance identity). Karakida Thms 1+4 kept
  as context/benchmark only, quoted within their real hypotheses.
- Lemma statement now carries an explicit head hypothesis (dense head
  over Theta(M) penultimate features with Theta(1) second moments) —
  covers per-pixel dense heads on MLP/CNN/ViT; "fixed depth =>
  Theta(M^2)" false implication corrected to a conditional; Kaiming
  wording unified (A56); dependency lists synced across statement,
  proof, restatement, combine step.
- Scaling story unified across 04/07/09/supplement (A6): SpatialMLP
  has C_g = Theta(M) so prediction is slope 1.0 vs C_g; measured
  0.70 +/- 0.05 reported as sublinear UNDERSHOOT (open discrepancy),
  sqrt(C_g)/slope-0.5 reserved for Theta(M^2) families. 07 relabeled
  to loss-Hessian blocks (A24 partial), SpatialMLP-only + ~3 orders
  of magnitude (A53 partial).
- Opportunistic: A39 (false <=1/4 softmax bound corrected), A63
  (draft-history meta-commentary made impersonal), A14 partial (02's
  Schur/condition-number sentence -> D_curv framing), softmax
  "Hessian" -> "Jacobian".
- Verification: round 1 found 2 major (Thm 4 misapplication; scope
  mismatch) + 5 minor; all fixed. Round 2 re-verification
  (wf_9d7f7921-a7f) on the witness proof + cross-file consistency.
- Paper compiles clean; only remaining warning is fig:exp11_paired
  (A10, Phase P5).

---

## 2026-08-17/18 — Fixes 1-4C applied + full 14-agent audit

**Fixes applied and pushed** (commits f0b81f9, 1ee6dff, cc455ee, 594a262):
- Fix 1: Theorem 1 restated with width-linear / sqrt(C_g) scaling.
- Fix 2: Lemma 4.3 generalized (later superseded).
- Fix 3 (Option C per external reviewer): Theorem 1 reformulated as
  module curvature disparity D_curv = lambda_max(G_pp)/lambda_max(G_tt)
  >= c1*M; Lemma 4.3 shrunk to operator-norm cap; kappa retired.
- Irrelevance-vs-starvation Remark added (honest hedging of Exp 1.3).
- Fix 4C (per external reviewer, rejecting my flawed 4A): Theorem 2
  fully rewritten as Finite-Time Spectral Starvation with polynomial
  residual envelope (Soudry-compatible), COMPLETE main-text proof by
  integration; classical-remedies subsection; Exp 1.3 honesty pass.

**Full-paper audit** (wf_0d0edeef-185; 9 finders, dedupe, 3 adversarial
verifiers, synthesis; ~1.7M subagent tokens across retries):
- 64 deduplicated findings: 11 BLOCKER / 19 MAJOR / 23 MODERATE /
  11 MINOR. Zero refuted under adversarial verification.
- Full table + 14-step fix order: REMAINING_WORK.md (committed).
- Most consequential: supplement Lemma 4.1 proof still derives the
  retired Omega(C_g) with fabricated Karakida theorem numbers (A1/A2);
  "Berkeley Manifold" still in bib (A3); C_g/C_f = 325 is wrong, real
  ~108 and unreconciled with companion's 18.3M (A4); the sqrt-scaling
  "agreement" story is WRONG for SpatialMLP whose C_g = Theta(D) —
  measured 0.70 UNDERSHOOTS the correct slope-1.0 prediction there
  (A6); Section 8 has 4 table rows with NO artifact backing, one
  contradicting the companion paper (A7), and an ordering claim its
  own table falsifies (A8); Exp 1.2 headline numbers irreproducible
  from artifacts (A19); Fisher-z t=14.65 confirmed pseudo-replication
  (A21); Exp 1.1 code measures full Hessian blocks, not GN blocks
  (A24); paper ~3x over Nature MI format (A11) — venue decision
  needed (auditor recommends ICLR 2027 as primary).

**Decision pending (user)**: venue (Nat MI restructure vs ICLR 2027 vs
Nature Communications) — gates the structural work; and A6 decision
(rerun Exp 1.1 with a Theta(M^2) architecture vs honest reframe).

**Next**: Step 2 integrity batch (bib author fix, Karakida proof
rewrite, fabricated-attribution removal) — independent of all
decisions and experiments.

---

## 2026-08-13 — Three-LLM external review: strategic pivot

**Done**:
- User ran an independent-soundness prompt through three LLMs (Claude,
  ChatGPT, Gemini) on the compiled main.pdf.
- All three converged on the same major theory issues. Not a soft
  review.

**Reviewer-converged theory holes**:
- (1) **λ_max = Ω(C_g) is a √-factor too strong.** Karakida's actual
  result is λ_max = Θ(width M), not Θ(params). For fixed-depth CNN/ViT
  where params ~ M², the correct bound is λ_max ~ √C_g. Our own
  Exp 1.1 empirical slope 0.70 sits in this √-supporting regime
  (log 74 / log 510 ≈ 0.69).
- (2) **Cauchy min-side interlacing step is invalid.** Counterexample:
  G = [[1,1],[1,1]] has λ_min⁺=2; principal submatrix A=[1] has
  λ_min⁺=1; "λ_min⁺(G) ≤ λ_min⁺(A)" fails 2 ≤ 1. Cauchy interlacing
  is about ordered eigenvalues including zeros; deleting zeros is not
  a Cauchy statement. The transfer step in our combine step is broken.
- (3) **Kronecker factorization J_θᵀJ_θ = AᵀA ⊗ Σ_X requires A_i ≡ A
  across pixels** — fails for real spatial CNN/ViT because output at
  pixel i depends on Z_j at neighbours (convolutional/attention mixing).
- (4) **Lipschitz assumption used on wrong derivative.** A2 bounds
  gradients w.r.t. φ; the proof uses ||∂ŷ/∂Z||_op ≤ L, different
  constant. Also input-Lipschitz of g_φ generally grows with width, so
  not C_g-independent.
- (5) **ε in Lemma 5.1 is zero** because H_θθ is rank-deficient. Fix:
  use λ_min⁺ consistently.
- (6) **Theorem 2's finite ∇L=0 contradicts CE dynamics** (Soudry
  2018: loss decays 1/t, predictor diverges to max-margin). Need L2
  regularization or directional restatement.

**Citations/arithmetic fixes needed**:
- "Berkeley Manifold" hallucinated author in Berisha 2019 entry.
- Karakida theorem numbers (3.1/4.2/5.1) don't exist; real numbers
  are Theorems 1, 3, 4.
- Horn-Johnson §1.20 for Haynsworth is wrong; correct is §4.5.P or
  §7.7.
- C_g/C_f ≈ 325 vs real-data ≈ 108 mismatch.
- Fisher-z aggregate t=14.65 is implausibly large; likely
  pseudo-replication.

**Novelty overclaims to walk back**:
- Huang et al. ICML 2022 "modality competition" already proves joint
  multi-modal training under-uses modalities. Distinct mechanism from
  ours but "the first rigorous explanation" overclaims.
- Chizat-Bach 2019 lazy training + Kirichenko 2023 DFR explain
  "frozen features > joint" differently. Prescription "follows from
  Theorem 2" should be softened to "consistent with."
- µP / Tensor Programs (Yang & Hu 2021) formalizes width-dependent
  gradient imbalance — bears on Prop 6.1 √(C_f/C_g) claim.

**Strategic decision** (user):
- Keep Nature MI target. PhD ends in ~7 months so Path C (empirical
  paper first, theory later) is too slow.
- Target 8-10 page main text + 15-20 page supplement, "short striking
  and concrete like old ML papers." Nothing like 50 pages.
- Willing to invest hours; but wants explanations at linalg 1+2 level
  so I can supervise.

**Key insight during the debrief**:
- The √C_g correction is actually a STRENGTH not a weakness. Our
  empirical slope 0.70 matches the corrected theorem better than the
  original one. "Theory predicts √, we measure 0.70, agrees with
  finite-width upper corrections."
- The per-pixel Kronecker case is a corollary; the general spatial
  bound uses residual-dimension rank ≤ H·W·(N_cls − 1), still
  C_g-independent. Spatial-spectral claim survives.

**Revised Path A** (35-45 hr total, spread over 10-15 sessions,
targeting ~2 weeks):
- Fix (1): restate Theorem 1 as κ_eff ≥ Ω(√C_g / C_2)
- Fix (2): rewrite Lemma 4.3 for general spatial g_φ using
  residual-dimension rank argument
- Fix (3): repair Cauchy → range-space transfer via Schur/pseudo-inverse
- Fix (4): reformulate Theorem 2 with L2 regularization or directional
- Fix all citations, arithmetic, novelty positioning
- Then finish Theorem 2 supplement proof, Exp 1.7/1.8, Section 8

**Next session**: Fix (1) + Fix (2) together — restate Theorem 1 with
√-scaling AND generalize Lemma 4.3 to spatial g_φ. Explained
pedagogically at linalg 1+2 level so the user can supervise each step.

---

## 2026-07-29 — Lemma 4.3 draft + combine step (later reviewed and found flawed)

**Done**:
- Walkthrough Q6 answered: λ_min⁺(J_θᵀJ_θ) = μ_r · σ_S via Kronecker
  spectral identity. Both factors C_g-independent.
- Wrote full LaTeX Lemma 4.3 proof (6 steps) in supplement.tex
  covering: Schur/Cauchy, Kronecker factorization,
  rank ≤ N_cls, spectral identity, μ_r ≤ L², assembly.
- Wrote combine step joining Lemmas 4.1, 4.2, 4.3 into Theorem 1's
  κ_eff ≥ c_1 · C_g/C_f - c_2 bound.
- Section 3: added Remark on effective condition number κ_eff =
  λ_max / λ_min⁺, since the joint GN is rank-deficient whenever
  N_cls < K.
- Fixed pre-existing paper compilation blocker: \thetap / \phip macros
  used \bm{\theta} which broke subscripts; wrapped \boldsymbol in
  extra braces. Paper now compiles to 436 KB PDF.
- Added Ghorbani 2019 + Magnus-Neudecker 1988 to references.bib.
- Rewrote muddled SK-factor manipulation in main-text Theorem 1 proof.
- Committed as 96db216; pushed main.pdf as b401969.

**Session-later reviewer flags** (see 2026-08-13 entry above):
- λ_max scaling exponent claim (linear instead of √) is wrong.
- Cauchy min-side transfer is invalid.
- Kronecker requires A_i ≡ A (not true for real CNN/ViT).
- Lipschitz assumption conflated (A2 is on φ, we used on Z).
- Karakida theorem numbers cited don't exist.
- Horn-Johnson Theorem 1.20 attribution suspicious.

**What survives**:
- The walkthrough intuition — bottleneck kills gradient — is correct.
- The Kronecker corollary works for per-pixel classifiers.
- The κ_eff definition is standard and fine.
- The main-text compile fix stays.

---

## 2026-07-21 to 2026-07-29 — Lemma 4.3 interactive walkthrough

**Done** (multiple short sessions with breaks):
- Set the scene: chain rule asymmetry
  ∂L/∂θ = (∂L/∂ŷ) · (∂ŷ/∂Z) · (∂Z/∂θ) has one more factor than
  ∂L/∂φ.
- Q1: extra factor A = ∂ŷ/∂Z represents classifier sensitivity to
  bottleneck.
- Q2: rank(A) ≤ min(rows, cols) = min(N_cls, K) = 2 in synthetic.
- Q3: rank(J_θ) ≤ rank(A) by matrix product rank rule.
- Q4: J_θᵀJ_θ has ≤ 2 nonzero eigenvalues out of C_f ≈ 1024 → 1022
  exact-zero eigenvalues.
- Q5: κ = ∞ trivially; need κ_eff = λ_max/λ_min⁺.
- Q6: smallest nonzero eigenvalue = μ_r · σ_S via Kronecker — neither
  factor depends on C_g.
- Established Lemma 4.3 intuition end-to-end. All these arguments
  survive the reviewer feedback; the FORMALIZATION is what has holes.
- User took breaks in the middle to refresh linear algebra via 3b1b
  videos. Explanations tuned to visual/geometric intuition.
- Follow-up conceptual questions: PSD matrix, rank of MᵀM, why
  eigenvalue decomposition matters, Cauchy interlacing intuition,
  why Σ_X shows up.

---

## 2026-06-30 to 2026-07-01 — Exp 1.6: Spectral Decoupling comparison + gap-of-time

**Done 2026-06-30**:
- Wrote exp1_6.py: 120 runs at D=256 (3 conditions × 4 noise × 5 seeds
  with 4-λ sweep on SD).
- Two workflows: research (7 parallel agents) + adversarial review
  (3 skeptics + synthesis). Caught 6 real BLOCKERs before launch.
- Result: joint vs frozen Wilcoxon T=6 p<0.0001 (41% collapse
  reduction); joint vs SD(best λ) T=88 p=0.55 (no effect).
- Committed + pushed as ed17f78.
- Verdict: Pezeshki SD does NOT reduce the two-timescale collapse
  that our freeze prescription targets.

**Done 2026-07-01**:
- Paper status snapshot for user: 65% to defensible submission.
- Recommended next: Lemma 4.3 + combine (closes Theorem 1 proof).
- Launched Lemma 4.3 walkthrough (paused frequently for user breaks
  and linalg refreshers).

---

**Done**:
- Ran exp1_4_fw (144 runs, D=256 fixed, noise × lr × wd × 6 seeds).
- Spawned 3-skeptic + synthesis Workflow on the new correlations.
- Synthesis verdict: **"real_but_weak"** — capacity-independent
  signal exists and survives every adversarial control, but
  practical effect size is moderate.

**Key numbers (D=256, n=144)**:
- Raw r(egr_min, final_acc) = +0.344 (p = 2.5e-5)
- Partial r controlling for noise/lr/wd = +0.395 (larger, not smaller)
  → NOT a hyperparameter confound
- Within-bin r (seeds only): mean +0.584 across 24 cells, 23/24 strongly
  positive, Fisher-z aggregate p < 1e-13
- Partial r controlling for train loss = +0.335 (essentially unchanged)
  → EGR adds value beyond train-loss tracking
- ROC-AUC for "bad run" classification = 0.612 (real but modest)
- Best threshold: TPR 0.90 / FPR 0.57 → not deployable as standalone alarm

**Section 6 updated**:
- Removed "capacity-aware diagnostic" framing.
- New structure:
    (i)  pooled correlation, capacity confounded (r = -0.78 but partial = -0.17)
    (ii) fixed-capacity test (r = +0.34, survives all controls)
    (iii) practical interpretation (AUC modest, use alongside train loss)
- Verdict: capacity-independent signal real; practically a complementary
  diagnostic, not a standalone alarm.

**Why this matters**: had we stopped at the pooled correlation,
reviewers would have crushed us with "this is just capacity in disguise."
The fixed-width follow-up + adversarial workflow give us a defensible,
honest claim.

**Next session**:
- Phase 2 W7: write Lemma 4.3 (Schur complement bound for lambda_min).
  This is the last load-bearing piece for Theorem 1's full proof.
- Then: Theorem 2 proof (Phase 3).

---

## 2026-06-30 — EGR fixed-width sweep + Karakida derivation drafted

**Done**:
- Wrote `code/experiments/exp1_4_fw.py` (fixed width D=256, sweeps
  noise × lr × weight_decay × 6 seeds = 144 runs).
- Launched in background — expected ~70 min.
- Drafted `study_notes/04_karakida_derivation.md` covering the
  asymptotic scaling argument for Lemma 4.1
  (`λ_max(J_φ^T J_φ) = Ω(C_g)`):
  - Karakida 2019 framework recap (mean-field FIM)
  - FIM ↔ Gauss-Newton conversion factor for CE
  - Per-architecture scaling (MLP / CNN / ViT) — all give λ_max ∝ C_g
  - Honest explanation of why our empirical slope is 0.7 not 1.0
    (finite-width corrections, bias parameters, CE residual)
  - 5-step proof outline for the supplement
- Lab talk: pushed to GitHub at
  https://github.com/kjdziuba/spectral_shortcut_theory; design brief
  in `presentation/DESIGN_BRIEF.md` is the handoff for design tool.

**Next session**:
- Read the fixed-width sweep result when it lands (notification).
- Run adversarial verification on the new correlation (workflow).
- If within-bin signal survives → upgrade Section 6 to
  "capacity-independent diagnostic"; otherwise leave the
  "capacity-aware" framing.
- Then: 2-hour focused session writing the Lemma 4.1 proof into
  `paper/sections/supplement.tex` using the Karakida derivation.

---

## 2026-06-29 — Exp 1.4 + adversarial verification + Pezeshki session

**Done**:
- User started reading Pezeshki 2021. Captured notes + follow-ups in
  `study_notes/02_pezeshki_notes_and_followups.md`:
    1. Add Spectral Decoupling as an experimental condition (Exp 1.6).
    2. Cite Pezeshki Sec 4 / App B robustness claim to avoid running
       those ablations ourselves.
    3. Prepare 3-4 slide presentation for spectroscopy people (TODO).
    4. NTK measurements not necessary for the main theorems.
- Wrote `study_notes/03_hessian_block_algebra.md` walkthrough — block
  matrix decomposition, Cauchy interlacing, Schur complement, Karakida
  scaling, combining into κ(G) ≥ Ω(C_g/C_f).
- Ran Exp 1.4 v1 (early-window EGR only) — got NEGATIVE correlation
  r=-0.56. Identified confound: capacity drives both.
- Iterated to Exp 1.4 v2 with multiple EGR windows (early/mid/late/min/depth).
- Headline pooled correlations: r(egr_depth, final_acc) = -0.78,
  Spearman -0.81.
- **Spawned adversarial verification workflow** (3 skeptics + synthesis):
    - Reviewer 1 (within-capacity): VERDICT = CONFOUND. Partial
      correlation controlling for log(width) drops to -0.17 (NS).
      EGR depth almost perfectly tracks width (r=0.90).
    - Reviewer 2 (depth vs min): VERDICT = DEPTH_IS_REAL. Both
      egr_early and egr_min contribute independent signal; the gap
      isn't just a min restatement.
    - Reviewer 3 (vs train loss): VERDICT = ADDS_VALUE. Partial r
      controlling for train_loss = -0.572 (significant).
    - Synthesis: VERDICT = MODERATE_DIAGNOSTIC. EGR depth is a
      "capacity-aware" diagnostic, not capacity-independent.
- Inserted honest synthesis paragraph into Section 6 with full caveats
  and a TODO marker for the follow-up at fixed width.

**Key insight**: Workflow's adversarial verification caught the same
type of confound that v1 had. Without it we might have written "EGR
depth is a strong real-time diagnostic" and gotten torn apart by
reviewers. The honest framing is "capacity-aware diagnostic — useful
when comparing comparable-width models, not yet a capacity-independent
predictor."

**Decisions**:
- Section 6 now has explicit caveats. Follow-up Phase 5 work: rerun
  at fixed width varying noise/lr/reg to test residual within-bin signal.
- Spectral Decoupling experiment to add as Exp 1.6 eventually.

**Next session**:
- Continue with Hessian algebra proof writing (Phase 2 W5-W7).
- Or: kick off the fixed-width EGR follow-up to upgrade the diagnostic.

---

## 2026-06-27 — Experiment 1.3 HEADLINE RESULT

**Done**:
- Identified design flaw in initial Exp 1.3: spatial info lived only
  in labels (label-position correlation), not in X. CNN couldn't
  extract pure position info during 150-epoch training. spatial_only
  baseline was stuck at chance across all three calibration modes.
- Fixed data generator: added per-position fixed spatial signature
  along v ⊥ u (orthogonal direction in R^S). Spatial signal now
  manifests directly in X content.
- Preserved v1 broken-design results as `_brokendesign` files.
- Re-ran exp1_3 with fixed data.

**Final results (CNN D=256, 3 seeds)**:

Bayes mode (alpha=0.82, beta=18.75):
- spectral-only acc = 0.53 (CNN training)
- spatial-only  acc = 0.53 (CNN training)
- frozen        acc = 0.61
- joint         acc = 0.48 (BELOW CHANCE)
- **shortcut gap = 0.13** (frozen - joint)

NTK mode: shortcut gap = 0.04
Margin mode: shortcut gap = 0.09 but poor calibration quality

**WINNER: Bayes mode** -- combined score 1.26 vs ntk 0.88 vs margin 0.82.

**The headline finding**:
Joint training collapses BELOW the spectral-only baseline (0.48 vs
0.53). This means end-to-end training doesn't just fail to combine
the two pathways -- it actively destroys spectral information that
the model demonstrably can use when given spectral-only data. This
is the direct empirical demonstration of the spectral-shortcut
conjecture from Theorem 2.

- Updated Section 7 with full Exp 1.3 write-up and calibration mode
  comparison table.

**Paper state at end of session**:
- All sections 1-9 have substantive drafts.
- All three experiments empirically validated:
    Exp 1.1: kappa scales monotonically (lambda_theta flat, lambda_phi grows)
    Exp 1.2: EGR collapse depth monotonic in capacity (CNN)
    Exp 1.3: joint training collapses BELOW single-modality baseline
- Real-data F1 table connects synthetic to FTIR/QCL findings.
- Discussion section with prescriptions, limitations, open problem.

**Next session**:
- Begin Phase 2 (Theorem 1 proof details, W5-W7).
- Read Pezeshki 2021 to build intuition for Theorem 2 proof.
- Consider running Exp 1.4 (EGR as predictor) to support Section 6.

---

## 2026-06-26 (night) — Experiment 1.3 designed; Sections 7, 8, 9 drafted

**Done**:
- Built `synthetic/calibrate.py` with 3 modes (Bayes, NTK, Margin).
  Fixed spatial-only baseline to use position-majority class (the
  correct primitive given our data design — position info lives in
  labels, not in X).
- Bayes calibration successful: alpha=0.82, beta=18.75 gives
  spectral-only acc=0.75, spatial-only acc=0.75 (calibrated equal).
- NTK and Margin modes can't calibrate spatial side for this data
  (no position info in X) — will document as a finding.
- Wrote `experiments/exp1_3.py`: 36 runs (3 modes × 4 conditions ×
  3 seeds), CNN D=256. Currently running in background.
- Drafted Section 7 (Experiments) with Exp 1.1 + Exp 1.2 results.
- Drafted Section 8 (Real-Data Validation) with FTIR/QCL F1 table
  connecting synthetic to real-data findings.
- Drafted Section 9 (Discussion) with prescriptions, limitations,
  Spatial Dominance Conjecture as open problem, beyond-spectroscopy
  applications.
- Added bib entries for Huang 2022, Saxe 2011, Rahimi-Recht, Bhatia,
  Horn-Johnson, Lyu-Li, Gunasekar, Chen GradNorm, Coil-Cheney,
  Bozzo 2024.

**Current paper state**:
- Sections 1-9 ALL have substantive drafts.
- TODOs are limited to actual proof details (Phases 2-3, W5-W10) and
  Experiment 1.3 final results.
- Supplement and abstract are stubs (per plan, written last).

**Next session**:
- Read Exp 1.3 results, pick best calibration, finalize Section 7.
- Begin Phase 2 (Theorem 1 proof details, W5-W7).
- User to read Pezeshki 2021 to build intuition for Theorem 2 proof.

---

## 2026-06-26 (late+) — Experiment 1.2 v4: CNN extended + ViT universality

**Done**:
- Added SpatialViT to `synthetic/models.py` (2-layer pre-LN ViT, ReLU
  activation for Hessian compatibility).
- Wrote `experiments/exp1_2v4.py` running 42 conditions (CNN widths
  {16,64,256,1024} + ViT widths {64,128,256}, joint vs frozen, 3 seeds).
- Drafted Section 1 (Introduction) properly.
- Drafted Section 2 (Related Work) with all 6 subsections.

**Findings (42-run sweep)**:

CNN sweep:
- EGR collapse depth scales monotonically with capacity (the headline
  Theorem 2 result):
    D=16   drops then recovers to ~0.4
    D=64   stable around ~0.25
    D=256  drops to ~0.10
    D=1024 drops to ~0.08
- Joint training final test loss diverges at high capacity:
    D=1024 joint test loss = 2.3, frozen test loss = 1.2
- Test accuracy gap exists but is small (~2-3 points) — joint overfits.

ViT sweep:
- Train loss reaches 0.01-0.03 at all widths — full memorization regime.
- Test loss is catastrophic (2.5-5.0) — joint training fails harder than CNN.
- EGR pattern INVERTED: rises with capacity rather than falling.
  This is the memorization regime where both gradient norms approach
  noise floor. Theorem 2's prediction holds in spirit (joint training
  fails) but the EGR observable loses signal.
- Frozen variants still beat joint at every width.

**Decisions**:
- For the paper: lead with CNN as the primary demonstration of Theorem 2.
  Note ViT as a different failure mode (memorization regime) where
  the EGR observable becomes uninformative but the overall prescription
  (freeze) still holds.
- Will mention in discussion that EGR diagnostic needs to be used
  before full memorization; otherwise the ratio loses meaning.

**Next session**:
- Move to Experiment 1.3 (equal-information killer test).
- Design the data calibration carefully (Shannon I(y; spectral) =
  I(y; spatial) via signal-strength tuning).

---

## 2026-06-26 (late) — Experiment 1.2 dynamics, three iterations

**v1 (Adam, SpatialMLP, noise=0.1):**
- Trained joint at D ∈ {32, 128, 512, 2048}, 3 seeds each, 150 epochs.
- Observed striking test-accuracy COLLAPSE at D=2048: peaks at ~70%
  early, then decays to ~50% by epoch 150 while train loss continues to
  drop. Two seeds at D=512 also collapse, one holds.
- EGR did not show dramatic decay (Adam's per-param normalization
  masks the natural gradient asymmetry).

**v2 (SGD+momentum, SpatialMLP, noise=0.05) — diagnostic miss:**
- Joint vs frozen converged to nearly identical accuracies (~75%).
- The collapse from v1 disappeared.
- ROOT CAUSE: SpatialMLP has zero spatial receptive field (per-pixel
  MLP). It has no spatial pathway for a "spatial shortcut" to exploit.
  The whole framing of "joint training discovers spatial shortcuts"
  cannot be demonstrated without 2D spatial mixing in g_phi.

**v3 (Adam, SpatialCNN with 2x 3x3 convs, noise=0.1) — the right setup:**
- Trained joint AND frozen at D ∈ {16, 64, 256}, 3 seeds each.
- Frozen beats joint at every capacity; gap widens with D.
  D=256: joint acc 54%, frozen acc 57%; joint test loss 1.05, frozen 0.80.
- Joint test accuracy at D=256 peaks ~70% then COLLAPSES to ~52%.
  Train loss meanwhile drops to 0.29 — classic shortcut overfit.
- EGR trajectories show the predicted capacity-monotonic collapse:
    D=16:  drops then recovers to ~0.4
    D=64:  drops to ~0.25 and plateaus
    D=256: drops to ~0.2 then keeps decaying to ~0.1
  This matches Proposition prop:egr_monotonic.

**Headline plots ready for paper drafts:**
- exp1_2v3_cnn_joint_vs_frozen.png — joint vs frozen by capacity
- exp1_2v3_cnn_egr_overlay.png    — capacity-monotonic EGR collapse
- exp1_2v3_cnn_joint_D256.png     — 4-panel full mechanism at D=256

**Decisions**:
- SpatialMLP retained as "no-spatial-mixing baseline" for ablations.
- SpatialCNN is the canonical architecture for the dynamics figures.
- Adam is the realistic optimizer for the literature contrast;
  SGD ablation can stay in supplement.
- noise=0.1 is the "interesting" regime (CE doesn't fully saturate,
  shortcuts emerge); we'll also report noise=0.05 in supplement to
  show the regime where joint training "looks fine."

**Next session**:
- Push SpatialCNN widths up to D=1024 to test whether the gap
  continues widening, or saturates somewhere.
- Add SpatialViT for the third architecture in our universality story.

---

## 2026-06-26 (eve) — Experiment 1.1 scaled (5 seeds, D up to 8192)

**Done**:
- Extended widths to D ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}.
- 5 seeds per D (5×10 = 50 measurements).
- Added paired eigenvalues plot (λ_θ and λ_φ on same axes).

**Findings**:
- λ_θ stays flat at ~0.02 across the full D range — confirms the spectral
  block does not grow because C_f is fixed. The pathology is entirely on
  the spatial side.
- λ_φ grows 74× (0.62 → 46) as C_g grows 510×.
- κ grows from 40 → 2,407 as C_g/C_f goes from 0.3 → 152.
- Fit slope is now 0.70 (up from 0.48 with the small range).
- At C_g/C_f ≈ 152 (close to real architecture's 325), κ ~ 2,400 —
  loss landscape is severely ill-conditioned.

**Decisions**:
- The "spectral block flat" finding is a HEADLINE result. Add paired
  plot as the primary figure for Theorem 1 in the paper.
- Will further push D to test asymptotic slope (whether it reaches 1)
  but the qualitative claim is firmly established.

**Next session**:
- Run Experiment 1.2 (training dynamics) — should give cleaner result.

---

## 2026-06-26 (PM) — Sections 4–6 drafted + Experiment 1.1 first run

**Done**:
- Drafted Section 4 (Theorem 1: Hessian Capacity Bound) — formal
  statement, 3 supporting lemmas (Karakida scaling, Cauchy interlacing,
  Schur complement), proof skeleton with TODO markers for Phase 2.
- Drafted Section 5 (Theorem 2: Two-Timescale Spurious Convergence) —
  formal statement, 3 supporting lemmas (timescale gap, fast manifold,
  residual decay), proof sketch combining Borkar + Pezeshki.
- Drafted Section 6 (EGR) — initial-value proposition, monotonicity
  proposition, threshold criterion, implementation reference.
- Workflow built infrastructure in parallel: synthetic/data.py,
  synthetic/models.py, hessian/eigenvalues.py, egr/callback.py,
  experiments/exp1_1.py, experiments/exp1_2.py. Smoke test passed.
- Fixed signature mismatch in exp1_1.py (top_eigenvalue_block API).
- **Ran Experiment 1.1 (capacity ablation)** on synthetic data:
    D ∈ {16, 32, 64, 128, 256, 512}, C_f = 1024, C_g ∈ [306, 9730]
    κ ranges from 23 to 464.

**Findings**:
- Direction confirmed: λ_φ grows with C_g, κ grows with C_g/C_f.
- Scaling exponent α ≈ 0.48 (predicted 1 from Karakida asymptote).
- κ already exceeds 400 at modest C_g/C_f ≈ 10 — landscape is
  pathological even far from the real-architecture regime
  (C_g/C_f ≈ 325).
- Need: scale D up to 2048+, average over seeds, run Exp 1.2 dynamics.

**Decisions**:
- Theorem 1 statement will be reframed as "monotonic growth with
  measurable exponent" rather than tight Ω(C_g/C_f).
- Cite Karakida for asymptote, document empirical finite-size α.
- Add capacity ablation up to D=8192 as next step.

**Next session**:
- Run Experiment 1.2 (dynamics) — should give clean qualitative result.
- Scale Experiment 1.1 to larger D + 5-seed averaging.
- User reads Pezeshki when ready.

---

## 2026-06-26 (AM) — Section 3 (Setup) drafted

**Done**:
- Walked through model setup, chain rule, Jacobian, Hessian conceptually
  - Simplified to C=1 (single-channel) for clean theory; channel extension trivial
  - Confirmed scope: linear `f_θ` for proofs, nonlinear empirically
  - Confirmed CE-specific role: vanishing-residual property
  - Concrete toy example (3-dim spectrum, 2 features, 2 classes) showing
    Jacobian = V in linear case
  - 2D quadratic example showing condition number → timescale gap
- Added Experiment 1.5 (Spatial architecture ablation: linear vs CNN vs ViT)
  - Important because CNN-based pipelines report most spectral-collapse issues
- Saved full conversational explanations to `study_notes/01_setup_and_chain_rule.md`
  - Convertible to PDF for offline study
- Wrote Section 3 (`paper/sections/03_setup.tex`) in formal LaTeX
  - Four assumptions stated (linearity, regularity, sub-Gaussian, capacity gap)
  - Loss formula
  - Chain-rule decomposition with named factors
  - Block Hessian structure
  - EGR formal definition
  - Notation conventions

**Decisions made**:
- Default C=1 in theory; real-data experiments use d2 (matches field standard)
- Run ablations later with raw, d1, d2 as robustness check
- Run BOTH linear and nonlinear (small MLP) in synthetic experiments
- Add CNN to spatial-architecture ablation (Experiment 1.5)

**Next session**:
- Begin Phase 0 (Track A reading): Pezeshki 2021
- In parallel: I start drafting Section 2 Related Work — gradient starvation subsection

**Open questions**:
- None yet.

---

## 2026-06-08 — Project initialized

**Done**:
- Created directory structure at `/home/u37314kd/Projects/spectral_shortcut_theory/`
- Wrote `README.md`, `PLAN.md` (8-phase plan), this `PROGRESS.md`
- Set up LaTeX paper skeleton with section stubs
- Created reading note templates for 5 foundational papers
- Initialized git repository

**Next session**:
- Begin Phase 0, Track A: Start reading Pezeshki et al. (2021) "Gradient Starvation"
- Fill in `reading_notes/01_pezeshki_2021.md` as you read
- In parallel, I'll start drafting Section 2 (Related Work) based on what you note

**Open questions**:
- None yet.

**Decisions log**:
- Project lives separate from `spectral_tokenization/` to keep theory work clean
- LaTeX-first for proofs; markdown drafts in `proofs/` for early iteration
- EGR diagnostic will be released as standalone PyPI package
- Target ICML/NeurIPS theory track first, AISTATS as backup
