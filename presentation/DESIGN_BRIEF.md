# Design brief — "The Spectral Shortcut" lab talk

Hand this to Claude Design / another design tool. It contains the
story, the numbers, the metaphors, and pointers to every plot we
already have. The two PPTX files I tried to build (`v1` and `v2`) are
in this folder for reference but don't look professional enough — use
this brief instead, ignore the slides.

---

## 1. Context

### Who is this for
Spectroscopy lab colleagues. PhD students and postdocs working on
FTIR / QCL / Raman tissue classification. They know spectroscopy and
chemometrics very well. They are math-light: they understand "loss
function", "gradient descent", "overfitting" at an intuitive level,
but have not seen Hessians, NTK, condition numbers, etc.

### How long
~5 minutes presented; ~7 slides max. Casual lab-meeting tone, not
conference style.

### Why we're presenting
We are writing a theory paper that explains a paradox the lab has
noticed empirically: end-to-end training of spectral-spatial deep
learning models keeps underperforming versions where part of the
model is frozen. We want to (a) communicate the project exists,
(b) explain the core idea, and (c) hint at why it matters for them.

---

## 2. The one-line story

"In spectral-spatial deep learning, training the whole model
end-to-end causes the spectral part to silently fail. We have a
mathematical explanation for why, a real-time diagnostic to detect
it, and a one-line fix (freeze the spectral encoder)."

---

## 3. The story arc (slide by slide)

### Slide 1 — Title

**Concept:** Establish the project.

**Suggested text:**
- Title: *The Spectral Shortcut*
- Subtitle: *Why locking part of a model makes it better*
- Author + lab meeting label.

**Visual idea:** Could be a literal padlock icon over the word
"spectral", or just very clean typography. Nothing busy.

---

### Slide 2 — The paradox (motivation)

**Concept:** Show, without explanation, that the same model architecture
gives radically different test F1 depending on whether we freeze part
of it. This is what the rest of the talk explains.

**Numbers from our companion empirical paper (real FTIR/QCL data, 5-fold CV):**

| Variant                                  | Test F1 |
|------------------------------------------|---------|
| End-to-end learned linear (joint)        | 0.675   |
| Frozen random spectral                   | 0.70    |
| Frozen PCA-128 spectral                  | 0.78    |
| Frozen pretrained spectral (Peak Windows)| 0.842   |
| Frozen pretrained spectral (MLP)         | 0.896   |
| Frozen pretrained spectral (Sliding Win) | 0.954   |
| Fine-tuned pretrained                    | 0.79    |

**Best visual treatment:** Two huge numbers side by side — **0.67**
(red, "trained end-to-end") and **0.95** (green, "froze half the
model") with a giant question mark or arrow between them. The detail
table can live in speaker notes; the slide should be just the two
numbers. Make it impossible to miss the gap.

**One-liner caption:** *Same model. Same data. Just trained differently.*

---

### Slide 3 — What's inside the model

**Concept:** Show the compositional pipeline visually. The audience
needs to viscerally understand the **capacity asymmetry** — one part
is ~300× bigger than the other.

**What the pipeline does:**
- INPUT: hyperspectral image, per pixel a 942-dim spectrum
- f_θ (spectral reducer): linear projection, 942 → 128. About 40,000 parameters.
- g_φ (spatial model): a Vision Transformer + decoder. About 13,000,000 parameters.
- OUTPUT: per-pixel tissue class

**Capacity ratio: C_g / C_f ≈ 325**

**Best visual treatment:** Pipeline flow diagram. Make the f_θ box
*literally tiny* and the g_φ box *literally huge* so the size ratio
is obvious without reading any numbers. Add a small spectrum
illustration at the input and a tissue-map illustration at the output.

**One-liner caption:** *A small "squeezer" feeds a huge "brain" — and
the size gap is the whole story.*

---

### Slide 4 — How gradients flow back (the chain rule)

**Concept:** Without going into derivatives properly, convey that
the gradient that updates f_θ has to flow back through three
factors, and if any of them is zero, f_θ stops learning.

**The equation:**
```
∂L/∂θ  =  ∂L/∂ŷ   ·   ∂ŷ/∂Z   ·   ∂Z/∂θ
            ↑           ↑           ↑
         residual    Jacobian     input
        (the error)  (what the    (the spectrum
                     spatial      itself)
                     model wants)
```

**Best metaphor:** A relay race. Three runners pass the same baton
backward. If the first runner has nothing to pass (residual = 0),
the rest of the chain receives nothing — it doesn't matter how
good the next runners are.

**Best visual treatment:** Three colored "tokens" / "runners" in a
row, with arrows passing the gradient signal backward through them
to θ. Color them: red (residual), orange (Jacobian), green (input).
Big simple equation above; minimal labels below each token.

**One-liner caption:** *Three steps, in a chain. If the first one
drops to zero, nothing reaches θ.*

---

### Slide 5 — What goes wrong

**Concept:** The mathematical mechanism in one sentence.
Cross-entropy is greedy — it pushes the model's predictions to be
confident as fast as possible. When predictions become confident,
the residual goes to zero. After that, f_θ never receives a useful
gradient again.

**Best metaphor:** Same relay race as slide 4, but now the first
runner has dropped the baton. Slow-motion failure.

**Best visual treatment:** Reuse the relay layout from slide 4, but:
- baton at the feet of the first runner with a red X
- downstream runners are pale / desaturated
- arrows between them are dashed/faded
- big "NO UPDATE" label at θ

**One-liner caption:** *Cross-entropy fits the easy patterns first.
After that there's no error left to learn the hard ones from.*

---

### Slide 6 — The shortcut, in pictures (the two-moons analogy)

**Concept:** Borrowed from Pezeshki et al. (NeurIPS 2021). Map their
two-moons example to our tissue classification setting so the
audience can see why "easy" and "right" are different.

**The example:**
Two interlocking moon-shaped classes in 2D. A linear (straight)
boundary can separate most of them but misses the corners. A curved
boundary separates them perfectly. **Gradient descent finds the
linear boundary and never the curve.** Why? Because the linear
direction has a bigger eigenvalue in the kernel — it gives a faster
loss decrease — and once the residual is near zero, the curved
direction never gets any gradient signal.

**Map this to spectroscopy:**
- "easy line" = MORPHOLOGY. Tissue region size, boundaries,
  density contrast. Easy for a CNN to see.
- "right curve" = CHEMISTRY. Subtle peak ratios, amide bands,
  lipid signatures. Hard, but generalizes better and is what
  the model SHOULD be using.
- Joint training picks morphology; the chemistry never gets the
  gradient it needs to specialize.

**Best visual treatment:** Two-panel split.
- Left: a clean scatter plot of the two-moons (blue cancer dots,
  orange normal dots), with a red dashed straight line ("easy line")
  and a green curve ("right curve") overlaid. **DO NOT** redraw this
  yourself by hand; the existing
  [presentation/img/moons.png](img/moons.png) is fine as a starting
  reference but a designer should remake it cleaner.
- Right: a stylized illustration mapping the same idea to tissue —
  maybe a sketch of a tissue with regions vs a spectrum with peaks.

**One-liner caption:** *Gradient descent picks the easy answer
(morphology). The hard answer (chemistry) never gets a chance.*

---

### Slide 7 — The fix and why it matters

**Concept:** Three actionable take-homes for someone doing
spectral-spatial DL.

**The three prescriptions:**
1. **Freeze the spectral module.** Train it separately on a pixel-
   level task (or use a fixed PCA / wavelet basis), then freeze it
   and train only the spatial model on top. Removes the
   timescale gap by construction.
2. **Or anchor it.** Inject a fixed baseline (PCA, fixed wavelets)
   alongside the learnable spectral output. The spatial model gets
   a stable input even if f_θ drifts.
3. **Watch the EGR.** Effective Gradient Ratio
   = ‖∇_θ L‖ / ‖∇_φ L‖. When this number drops sharply during
   training, the spectral module is being starved. Use as a
   real-time diagnostic.

**Why it matters for the audience:**
- Every FTIR / QCL / Raman / hyperspectral paper that uses
  end-to-end joint training is leaving test F1 on the table.
- We can publish a one-line recipe ("freeze first") backed by
  rigorous math.

**Best visual treatment:** Three big icons in a row — padlock,
anchor, gauge — with one-line labels. No body copy.

**Closing one-liner (anywhere on the slide):** *Don't train
everything at once. The math says you can't.*

---

## 4. Plots and assets we already have

If the designer wants to embed real data into the talk, here are the
ones worth using and what they show. All paths are relative to
the project root.

### Plot A — the Hessian κ blow-up
**Path:** `results/exp1_1_paired_eigenvalues.png`
**Shows:** As we make the spatial model wider (D from 16 to 8192,
x-axis), the spatial-block top eigenvalue grows by 75× (blue line
going up). The spectral-block top eigenvalue stays flat at ~0.02
(red line, basically horizontal). The ratio of these two — the
condition number κ — climbs from ~40 to ~2,400.

**How to use:** Annotate the two curves in plain language:
- blue arrow: "spatial part keeps growing"
- red arrow: "spectral part stays flat"
- highlight a single point at D=8192 showing κ ≈ 2,400

### Plot B — the equal-information killer test
**Path:** `results/exp1_3v1_summary.png`
**Shows:** Three calibration modes side by side. In the "Bayes"
mode (left panel), the spectral pathway and spatial pathway are
calibrated to carry EQUAL information about the label. Result:
end-to-end joint training (red bar, 0.48) is *worse than chance*
and worse than either single-pathway baseline (green/purple bars,
~0.53). Frozen variant (blue bar, 0.61) wins.

**How to use:** Show only the Bayes panel (leftmost third) for
clarity. Bars labelled in plain language:
- joint training: 0.48
- frozen spectral: 0.61
- only spectral info available: 0.53
- only spatial info available: 0.53

### Plot C — gradient norms over training
**Path:** `results/exp1_2v3_cnn_grad_norms_D256.png`
**Shows:** Spatial gradient stays moderate (blue, ~0.05). Spectral
gradient (red) starts moderate but collapses by step 500 and never
recovers. Visualises the relay-drop concretely.

### Plot D — Two-moons illustration (reference only)
**Path:** `presentation/img/moons.png`
**Shows:** A passable but rough two-moons illustration with the
"easy line" and "right curve". A real designer should redraw this
more cleanly — bigger dots, smoother classes, nicer typography.

### Other plots in `results/` worth knowing about
- `exp1_2v3_cnn_egr_overlay.png` — EGR trajectories at 4 widths,
  showing capacity-monotonic collapse.
- `exp1_4_correlation.png` — EGR depth correlates with final
  accuracy across our sweep.

---

## 5. Numbers to keep on hand

For the speaker notes or for any text overlays:

- Real data: end-to-end 0.675 F1 vs frozen Slidewin 0.954 F1.
- Synthetic: Hessian condition number κ from 40 to 2,407 across the capacity sweep.
- Synthetic killer test: joint training 0.48, frozen 0.61, single-pathway baselines 0.53.
- Capacity ratio of our spectroscopy pipeline: C_g / C_f ≈ 325.
- EGR follow-up: pooled correlation r = -0.78 between EGR depth and test accuracy. Honest caveat: ~77% of that is capacity-driven, not a clean capacity-independent diagnostic yet.

---

## 6. Tone and style

- **Casual, conversational.** This is a lab meeting, not a
  conference talk.
- **Visuals over words.** Every slide should have a single
  dominant image. Text is supporting, not central.
- **Concrete examples over abstract math.** The relay metaphor and
  the two-moons example should carry most of the explanation.
- **Equations are okay** *only* if they read naturally as English
  (e.g., the chain rule on slide 4).
- **Plot less than 5 colors total.** Suggested palette:
  - navy: titles, structure
  - blue (#3D7DC7): spatial / data / first class
  - orange (#F58A1F): spectral / second class
  - green: "right" / success / freezing
  - red: "easy" / failure / unfrozen
  - grey: captions, secondary text

---

## 7. What I tried before (don't repeat my mistakes)

I built two attempts (`v1` and `v2`). Both have the problem of looking
AI-generated:

- v1 was too professional / corporate — too much text, no metaphors.
- v2 used custom matplotlib illustrations (relay runners, padlock
  icons) which look amateurish. The alignment is fine, but the
  drawings look like clip art.

The best parts to KEEP:
- The story arc (7 slides, this order).
- The relay metaphor for the chain rule.
- The two-moons example for the shortcut.
- The big "0.67 vs 0.95" framing on the paradox slide.

The parts to REDO:
- All custom illustrations should be remade by a designer with
  proper graphic-design taste.
- The pipeline slide should use cleaner iconography — maybe a
  literal spectrum (zigzag line), a literal small box, a literal
  large box, a literal tissue map.
- The runners on slide 4/5 should be replaced with something more
  visually elegant than three matplotlib stick figures.

---

## 8. Quick prompts you can give a design tool

If you're handing this off to Claude Design or similar, three
self-contained prompts:

### Prompt for the deck overall
> Build a 7-slide casual lab meeting deck called "The Spectral
> Shortcut". Audience: spectroscopy researchers without ML theory
> background. Tone: image-first, very few words per slide, friendly
> not corporate. Follow the slide-by-slide content in
> `presentation/DESIGN_BRIEF.md` exactly. Use the palette navy /
> blue #3D7DC7 / orange #F58A1F / green / red / grey.

### Prompt for the relay illustration (slide 4 / 5)
> Three abstract figures in a row passing a glowing object (the
> gradient signal) backward. Each figure is one color: red, orange,
> green. The signal starts on the far right with arrow flowing to
> the left, passing through each figure, ending at a labeled θ
> symbol on the far left. Clean modern flat illustration, not
> cartoony, not corporate clip-art. For slide 5, redraw the same
> composition but with the rightmost figure having dropped the
> signal — it lies on the ground with a red X — and downstream
> arrows are faded.

### Prompt for the two-moons illustration (slide 6)
> Clean scientific scatter plot. Two interlocking moon-shaped
> classes: blue points labeled "cancer cells" forming the upper
> moon, orange points labeled "normal cells" forming the lower
> moon. Overlay two decision boundaries: a red dashed straight
> line labeled "easy line" that misses the corners, and a green
> smooth curve labeled "right curve" that separates them cleanly.
> No grid, no axis labels. Big legend in lower right.

---

That's the brief. Hand this off, or use it as a checklist for
yourself when building the deck in PowerPoint / Keynote / Figma.
