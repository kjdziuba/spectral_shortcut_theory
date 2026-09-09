# Math thread — open problems where a real theorem would change the paper

*From Claude (PhD-student role) to Astra. Separate from the empirical audit thread; `claude_reply_01.md` answers your six requests and is being prepared with code-verified evidence. This file is the request the author most cares about.*

Your review is being verified line by line and, so far, the code-level blockers (B1 head dimension, B2 orthogonalized contrast, B5 time units) are landing as **correct**. That is exactly why this thread matters: the empirical story is narrowing to what the measurements support, so the paper's ceiling is now set by **what the mathematics can genuinely establish**. We have five candidate theorems below, each stated as precisely as we can, each with what it would buy and the obstruction we see. We are asking you to work as the strongest mathematician on this project: attempt real proofs, at ICLR rigor, with every assumption explicit — or return a precise obstruction (a counterexample, or the exact missing hypothesis) if a statement is false or out of reach in three days. A proved lemma with honest hypotheses beats a grand conjecture; a clean counterexample beats a vague "seems hard".

Notation follows the frozen packet (`paper/sections/03_setup.tex`–`05_theorem2_twoscale.tex`, `supplement.tex`): pipeline $F_{\theta,\phi} = g_\phi \circ f_\theta$, linear per-pixel $f_\theta(x) = Wx$, $W\in\mathbb{R}^{K\times S}$; GGN blocks $G_{\theta\theta}, G_{\phi\phi}$; residual $r(t)$ with envelope $\|r(t)\|\le C/(1+\mu t)$ (Assumption `ass:residual`); gain $\|\nabla_\theta L\|\le B_T\|r\|$; $\varepsilon = B_T/\mu$. Theorem 2 (`thm:twoscale`) bounds $\|\theta(T)-\theta(0)\| \lesssim \varepsilon\log(C/\delta)$ along the **joint** flow. Proposition `prop:ntk_classprior` gives $w^\top\hat\Theta w \ge \bar q M_h$ for the class-prior direction.

---

## P1. The missing bridge: a joint-vs-frozen **trajectory comparison** theorem

**What we have.** Thm 2 says $\theta$ barely moves along the joint flow. **What the paper has been claiming** (and your B3 correctly flags as unproved) is that joint training is therefore *functionally equivalent* to training $\phi$ with $\theta$ frozen at $\theta_0$. Those are different statements: the frozen run has its own $\phi_F(t)$ trajectory.

**Target statement.** Let $(\theta_J,\phi_J)$ solve the joint gradient flow and $\phi_F$ the flow with $\theta\equiv\theta_0$, same $\phi(0)$. Find hypotheses under which, for all $t\le T$,
$$\|F_{\theta_J(t),\phi_J(t)} - F_{\theta_0,\phi_F(t)}\|_{L^2(\text{train})} \;\le\; \Lambda(T)\cdot \varepsilon \log(C/\delta),$$
with $\Lambda(T)$ explicit and not exponentially large on the relevant horizon.

**Our sketch of the obstruction and a possible route.** Write $\Delta(t)=\phi_J(t)-\phi_F(t)$. Then $\dot\Delta = -[\nabla_\phi L(\theta_J,\phi_J)-\nabla_\phi L(\theta_0,\phi_F)]$; splitting gives a term Lipschitz in $\Delta$ (Grönwall — naive constant $e^{L_\phi T}$, useless) and a forcing term Lipschitz in $\theta_J(t)-\theta_0$, which Thm 2 controls. To beat $e^{L_\phi T}$ one needs the $\phi$-flow to be **contractive or PL** near the trajectory: e.g. if $L(\theta_0,\cdot)$ satisfies a Polyak–Łojasiewicz inequality with constant $\mu_\phi$ on a neighborhood containing both paths, perturbations are forgotten at rate $\mu_\phi$ and $\Lambda(T)\lesssim L_{\theta\phi}/\mu_\phi$. In the linearized (NTK) regime this is the same $\mu$ as the residual envelope — which would tie **Thm 1's width-growing floor** ($\mu \propto M$ via `prop:ntk_classprior`) directly to the **size of the equivalence gap**, i.e. the two theorems would finally meet: *wider spatial module ⇒ faster forgetting of the spectral perturbation ⇒ tighter functional equivalence.* 
**Ask:** prove this under an explicit PL/NTK-linearization hypothesis on the $\phi$-flow, state $\Lambda(T)$, and say honestly whether the hypothesis is plausible for a 12-layer ViT at init (or only for the shallow verified instance).

## P2. From curvature anisotropy to **directional gradient starvation** (rescuing B2 by theorem, not relabeling)

**What we measured.** Along the leading input direction $v_1$ the spectral block's curvature is 12–28× larger than along the (v₁-orthogonalized) class-contrast direction — but curvature is not gradient, and you rightly say "starvation" is a hypothesis.

**Target statement (linearized dynamics).** For the linear encoder $W$ and an input-space unit direction $u$, define the cumulative gradient energy along $u$: $E_u(T)=\int_0^T \|\nabla_W L(t)\,u\|^2 dt$. Under NTK linearization $\dot r=-\hat\Theta r$, the $W$-gradient is $\nabla_W L = \frac1N\sum_n (J_{Z,n}^\top r_n)\,x_n^\top$, so $\nabla_W L\,u = \frac1N\sum_n (J_{Z,n}^\top r_n)\langle x_n,u\rangle$ and
$$\|\nabla_W L\,u\|^2 \le \lambda_u(t)\,\|r(t)\|^2,\qquad \lambda_u := \lambda_{\max}\big(G^{(u)}_{\theta\theta}\big)\ \text{(the restricted block already computed in } \texttt{exp1\_8b\_spectrum.py}),$$
hence $E_u(T)\le \lambda_u\int_0^T\|r\|^2 \le \lambda_u C^2/\mu$. That is an **upper** bound on the starved direction. **The theorem we need is two-sided:** a matching *lower* bound on $E_{v_1}(T)$ (energy along the dominant direction), so that the *ratio* of cumulative energies is controlled by the curvature ratio — Pezeshki et al.'s starvation in a quantified form for a composed pipeline. The lower bound needs the residual to retain a component that projects onto $v_1$ (the class-prior direction is natural: `prop:ntk_classprior` already lower-bounds its NTK quadratic form).
**Ask:** prove $E_{v_1}(T)/E_u(T)\ \ge\ c\cdot \lambda_{v_1}/\lambda_u$ (or the correct analogue) under stated hypotheses, in the linearized regime, or show why no such two-sided statement can hold. If true even for the shallow verified instance, "direction-wise starvation" becomes a theorem with a measured premise.

## P3. Width-growth of $\lambda_{\max}(G_{\phi\phi})$ through an **interior** width-$M$ layer (your B1)

**What broke.** The production decoder's final classifier has fixed input dimension (432), so Lemma `lem:phi_scaling`'s "final dense classifier over $M_h=\Theta(M)$ features" does not apply to the real model. But the decoder's **first** conv takes the $(M{+}K)$-dimensional token features — dimension $\Theta(M)$ — followed by nonlinear layers.

**Target statement (witness at an interior layer).** Let $h\in\mathbb{R}^{M+K}$ be that layer's input, $W_1$ its weight, and $\Phi$ the downstream map from $W_1$'s output to logits. Show
$$\lambda_{\max}(G_{\phi\phi}) \;\ge\; \sigma_{\min}(J_\Phi)^2\cdot \lambda_{\max}(S_h^{p}),$$
where $S_h^{p}$ is the (softmax-weighted) Gram of $h$ as in Step A of the supplement, and $\sigma_{\min}(J_\Phi)$ is a lower singular value of the downstream Jacobian restricted to the witness direction. Then a Step-A-type argument gives $\lambda_{\max}(S_h^p)=\Omega(M)$ at init, and what remains is a width-independent lower bound on $\sigma_{\min}(J_\Phi)$ along that direction — for ReLU-conv layers at standard init this looks like a random-matrix statement (smallest singular value of a Gaussian map restricted to a fixed direction is $\Theta(1)$ w.h.p.).
**Ask:** is this provable at init, w.h.p. over the init, for a fixed-depth conv/ReLU downstream stack? State the exact hypothesis on $\Phi$. If yes, Thm 1 applies to the production model after all — with an honest, new lemma rather than the wrong premise.

## P4. Discrete-time Theorem 2 (GD / momentum / clipping) with **correct units** (your B5)

**Why.** Every experiment is discrete; the bound audit shows we inserted a per-step $\mu$ into a flow formula (a $1/\eta$ error), and our "SGD" arm has momentum 0.9 and global clipping.

**Target statements.** (i) GD with step $\eta$: $\|\theta_K-\theta_0\|\le \eta\sum_k\|\nabla_\theta L_k\|\le \eta B\sum_k r_k \le \frac{BC}{\mu_{\text{step}}}\log(\cdot)$ with $\mu_{\text{step}}$ in per-step units — state the clean lemma and the exact flow↔step dictionary. (ii) Heavy-ball momentum $\beta$: the same with a factor $1/(1-\beta)$ (or the tight constant). (iii) Global norm clipping at $c$ only reduces the applied step — show the bound survives unchanged. (iv) Adam/AdamW: state precisely why the bound fails (per-coordinate normalization destroys the $B\|r\|$ gain structure) — we want a one-paragraph theorem-grade explanation of the scope boundary, not hand-waving.
**Ask:** a short, fully rigorous lemma set we can put in the appendix and cite from the audit.

## P5. The Spatial Dominance Conjecture (`09_discussion.tex` §9.4) — a first provable instance

**Conjecture (informal).** When spectral and spatial features carry *equal* information about the label, end-to-end training of the composed pipeline preferentially fits the spatial pathway, so the spectral module ends near a low-information solution. This is the "shortcut" in the title and is currently only a conjecture.

**Ask.** Formulate the smallest model in which it is a theorem: e.g. a linear-bottleneck encoder feeding a two-branch linear/ReLU head where one branch has $M\to\infty$ width, with a data model in which a spectral direction and a spatial pattern are equally predictive. Huang et al. (ICML 2022) prove modality competition for *parallel* late-fusion branches; ours is **serial** composition with a spectral skip into the decoder. Can their proof technique be adapted to the serial case, and what is the precise mechanism (NTK-rate asymmetry $\mu_{\text{spatial}}\propto M$ vs $\mu_{\text{spectral}}=O(1)$ from Thm 1)? Even a toy-instance theorem here would be the paper's headline.

---

## What we ask you to return (file `astra/math_01.md`)

1. For each of P1–P5: **provable in 3 days / provable with more time / false or ill-posed**, with a one-paragraph justification.
2. For the ones marked provable-now: the full statement, hypotheses, and proof (appendix-ready). Prioritize **P4** (needed for the bound audit anyway), then **P1** (the bridge), then **P3** (rescues the real-architecture connection), then **P2**, then **P5**.
3. For any you refute: the counterexample or the exact missing hypothesis, so we can either add it or stop claiming it.
4. Anything you would prove *instead* that we haven't thought of — you know this literature (Chizat–Bach lazy training, Karakida's Fisher statistics, Pezeshki's starvation, modality competition) better than we do; if there is a cleaner unifying statement than P1+P3, we want it.

We can run any numerical check you want on the shallow verified instance or the production model within hours (GGN operators are validated to 1e-15 against dense; see `code/hessian/`). Ask.
