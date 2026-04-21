# Chapter 2 — Methodology: Writing Brief (v3)

**Target:** ~7 pages including Figure 2a
**LaTeX file:** `chapters/chapter2.tex`
**Skeleton version:** v3 (restructured to Leeds Final Report template)

> **Critical framing note:** Chapter 2 covers **WHY** design decisions were made. All implementation details (the **how**) are deferred to Chapter 3 (Implementation & Validation). Do not describe code, training loops, or dataset loading steps here — only justify the choices.

---

## Chapter-Level Intro Paragraph

One paragraph (3–4 sentences). State what the chapter covers: the problem formulation, baseline architecture rationale, rationale for each of the three improvements, ablation study design, and project management approach. Forward-reference Chapter 3 for implementation details and Chapter 4 for results.

**Transition out:** "The chapter begins by formally defining the recognition task before motivating the baseline architecture and each subsequent modification."

---

## 2.1 Problem Formulation

**Target length:** Short (~100 words, 2 paragraphs)

**Purpose:** Define the task precisely so the reader understands what the model is optimising. This is the formal anchor for all design decisions that follow.

**Paragraph structure:**

1. *Topic: Formal task definition.*
   Given a sequence of 2D joint coordinates of shape $(T,\,13,\,2)$, predict the action class label from 15 classes. The input is a fixed-length skeletal sequence produced after preprocessing (described in Chapter~3); the output is a single class index from the Penn Action label set.

2. *Topic: Evaluation objective.*
   The evaluation objective is to maximise top-1 test accuracy on the held-out Penn Action test set, subject to a lightweight parameter budget. Per-class accuracy is reported separately for exercise and non-exercise classes to assess domain-specific performance (see §4.2).

**[TBD] flags:** None — all values are fixed.

**Transition out:** "The baseline architecture is designed to satisfy this objective with minimal architectural complexity."

---

## 2.2 Baseline Architecture Design

**Target length:** Medium (~300 words, 4 paragraphs)

**Purpose:** Motivate every design choice in the baseline — not describe the code. The reader should understand *why* 6 blocks, *why* these channel widths, *why* this classification head, *why* fixed adjacency.

**Paragraph structure:**

1. *Topic: Why the 6-block ST-GCN was chosen.*
   The 6-block ST-GCN [1] is the foundational, well-understood graph convolutional model for skeleton-based action recognition. It is directly traceable to Yan et al. [1], documented in the community, and cleanly extensible — each of the three subsequent modifications targets a single component. The channel progression [32, 32, 64, 64, 128, 128] is chosen to balance representational capacity against Penn Action's small training set (~1,545 clips); heavier channel widths risk overfitting before the adaptive modifications are introduced.

2. *Topic: Rationale for the three-partition fixed adjacency strategy.*
   The spatial graph convolution operates on a fixed adjacency matrix constructed using Yan et al.'s three-partition strategy: self-loops, centripetal edges (joints toward the body centre), and centrifugal edges (joints away from centre) [1]. This partitioning provides a physically grounded inductive bias — joints closer to the action centre receive proportionally stronger aggregation — and serves as the stable structural prior that Improvement 2 (§2.4) augments with learnable components.

3. *Topic: Classification head design rationale.*
   Following the final ST-GCN block, a Global Average Pooling (GAP) layer collapses the temporal dimension without introducing learnable parameters, producing a 128-dimensional feature vector. BatchNorm and Dropout($p = 0.2$) are applied before the two fully connected layers (FC(128) → FC(13)). The conservative dropout rate of 0.2 is appropriate for a small dataset: aggressive dropout degrades convergence on limited data. The full baseline architecture is shown in Figure~\ref{fig:baseline-arch}.

4. *Topic: Forward link to Figure 2a.*
   Figure~\ref{fig:baseline-arch} shows the stacked 6-block structure, annotated with channel widths (32, 32, 64, 64, 128, 128), temporal resolutions (100, 100, 50, 50, 25, 25 frames), and the classification head. Temporal downsampling occurs at blocks 3 and 5 via stride-2 temporal convolution.

**Figures/tables required:**
- **Figure 2a** (`fig:baseline-arch`) — Baseline 6-Block ST-GCN Architecture. See diss-diagrams spec below.

**■ Figure 2a — Baseline 6-Block ST-GCN Architecture**
Vertical or horizontal stack of 6 labelled ST-GCN blocks. Each block annotated with: input channels → [GCN | TCN | BN | ReLU | Residual] → output channels. Channel widths (32→32→64→64→128→128), temporal resolutions (T=100,100,50,50,25,25), stride markers on blocks 3 and 5. Append classification head: GAP → BN → Dropout(0.2) → FC(128) → FC(13, softmax). Annotate input shape (N,2,100,13) and output shape (N,13). Use TikZ style keys: blue convblock, green bnblock, orange actblock, yellow addblock.

**Citations:** `yan2018spatial` [1]

**Transition out:** "This baseline is extended through three sequential modifications, each targeting a different source of inaccuracy."

---

## 2.3 Improvement 1: Skeleton Data Augmentation — Rationale

**Target length:** Medium (~250 words, 4 paragraphs)

**Purpose:** Justify the decision to add augmentation and each specific transform chosen. Do **not** describe implementation code — that belongs in §3.3.

**Paragraph structure:**

1. *Topic: Motivation — observed overfitting in the baseline.*
  With approximately 1,545 training samples across 15 classes, regularisation through data augmentation is the principled first intervention: it increases effective sample diversity without acquiring new labelled data [9].

2. *Topic: Five augmentation transforms selected.*
   Five transforms are applied online at training time only: random rotation, random scaling, Gaussian joint jitter, random temporal crop, and speed perturbation [9]. These are drawn directly from the literature without further tuning on the validation set, avoiding the risk of over-fitting the validation split during development. Validation and test sets use unmodified original sequences.

3. *Topic: Domain motivation for speed perturbation.*
   Speed perturbation is the single most domain-relevant transform for Penn Action's exercise classes. Exercise actions exhibit high intra-class cadence variation — fast squats versus slow squats share the same label but differ substantially in frame-count distribution. Speed perturbation directly targets this variability and is predicted to produce the largest accuracy gain within the augmentation bundle.

4. *Topic: Spatial augmentations preserve graph topology.*
   Random rotation and Gaussian joint jitter act independently on each frame's coordinate values. Neither modifies the adjacency matrix: joint connectivity is defined by joint identity, not by absolute coordinate position. The graph structure therefore remains intact, and augmented samples are valid inputs to the same ST-GCN blocks used for unaugmented sequences.

**Citations:** `shorten2019survey` [9], `xin2024enhancing` [9]

**[TBD] flags:** None — augmentation strategy is fully specified from the literature.

**Transition out:** "The second modification targets the graph topology rather than the input data."

---

## 2.4 Improvement 2: Adaptive Adjacency Matrices — Rationale

**Target length:** Medium (~250 words, 4 paragraphs)

**Purpose:** Motivate replacing the fixed adjacency with the learnable decomposition and honestly frame the expected negative result on small data.

**Paragraph structure:**

1. *Topic: Limitation of the fixed adjacency.*
   The fixed adjacency matrix encodes only physical bone connections. Exercise motions frequently involve strong correlations between non-adjacent joints — bilateral wrist coupling during push-ups, wrist–ankle coupling during jumping jacks — that a fixed skeleton graph cannot capture. An adaptive topology can learn these latent dependencies from training data [2].

2. *Topic: Adopted formulation — 2S-AGCN decomposition.*
   The 2S-AGCN three-component decomposition $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{B} + \mathbf{C}$ is substituted into the spatial graph convolution of every ST-GCN block [2]. $\mathbf{A}$ provides the frozen structural prior; $\mathbf{B}$ is a globally learnable $13 \times 13$ matrix initialised to zero that learns class-invariant implicit dependencies; $\mathbf{C}$ is computed per sample via two $1 \times 1$ convolution branches whose dot product is row-wise softmaxed, capturing instance-level co-activation. All other hyperparameters remain identical to the baseline to isolate the effect of the adaptive topology.

3. *Topic: Parameter overhead.*
   $\mathbf{B}$ adds 169 parameters per block (6 blocks = 1,014 additional parameters total). $\mathbf{C}$ adds two $1 \times 1$ convolution branches per block. The total overhead is modest relative to the baseline parameter count.

4. *Topic: Dataset-size caveat and anticipated negative result.*
   With approximately 1,545 training samples, the sample-adaptive component $\mathbf{C}$ has limited data from which to learn meaningful per-sample attention. This is a known limitation of the 2S-AGCN formulation on small datasets: the component that provides the most expressiveness is also the component most prone to overfitting. This motivates framing the Improvement 2 result as a well-documented negative finding if accuracy gains are modest or absent — not as an implementation defect (see §4.1 for analysis).

**Citations:** `shi2019two` [2]

**[TBD] flags:** None — parameter counts are derivable; negative framing is established.

**Transition out:** "The third modification extends the input representation from a single stream to four complementary modalities."

---

## 2.5 Improvement 3: Four-Stream Fusion — Rationale

**Target length:** Medium (~250 words, 4 paragraphs)

**Purpose:** Justify the four-stream approach: why different modalities, why score-level fusion, why fixed weights.

**Paragraph structure:**

1. *Topic: Motivation — complementary representations.*
   Different input representations encode qualitatively different aspects of motion. Absolute joint coordinates encode pose geometry; bone vectors (directed differences between connected joints) encode limb orientation and length; temporal differences of joints and bones encode velocity and acceleration dynamics. No single representation captures all three simultaneously.

2. *Topic: Adopted strategy — four independent adaptive ST-GCN models.*
   Four architecturally identical adaptive ST-GCN models (each equipped with the learnable adjacency $\hat{\mathbf{A}}$ from Improvement 2) are trained independently — one per modality: joint ($\mathbf{v}$), bone ($\mathbf{b}$), joint-motion ($\mathbf{m}^J$), and bone-motion ($\mathbf{m}^B$). This follows the MS-AAGCN strategy [3], which demonstrates a consistent 3–5% accuracy gain over a single joint stream on major benchmarks.

3. *Topic: Justification for score-level fusion.*
   Score-level weighted average fusion avoids jointly training a large multi-stream model, which would be infeasible given Penn Action's small training set. Independent training also enables per-stream accuracy analysis (§4.1), making the source of any accuracy gain interpretable. The final prediction is:
   \[\hat{y} = w_J\,p^{J} + w_B\,p^{B} + w_{JM}\,p^{JM} + w_{BM}\,p^{BM}\]
   with weights $w_J : w_B : w_{JM} : w_{BM} = 2:1:2:1$, normalised to sum to 1 [3].

4. *Topic: Justification for fixed literature weights.*
   The (2:1:2:1) weighting is taken directly from MS-AAGCN [3] without re-optimisation for Penn Action. Re-optimising fusion weights on the validation set introduces a risk of over-fitting to the small validation partition. Using literature defaults provides a reproducible, unbiased baseline for the fusion component; learned weights are identified as a future direction (§4.7).

**Citations:** `shi2020skeleton` [3]

**[TBD] flags:** None — weights are fixed at 2:1:2:1 from [3].

---

## 2.6 Ablation Study Design

**Target length:** Short–Medium (~200 words, 3 paragraphs)

**Purpose:** Define the four conditions and justify the sequential, single-modification protocol. This is the *design* of the evaluation — quantitative results appear in Chapter 4.

**Paragraph structure:**

1. *Topic: Sequential ablation rationale.*
   The experiment is structured so that each condition adds exactly one modification relative to the previous, isolating the marginal contribution of each improvement. This allows observed accuracy changes to be attributed cleanly, rather than confounded by simultaneous modifications.

2. *Topic: The four conditions.*

   | Condition | Description | Change from previous |
   |-----------|-------------|----------------------|
   | (1) | Baseline ST-GCN — fixed adjacency $\mathbf{A}$, no augmentation, single joint stream | — |
   | (2) | +Skeleton Data Augmentation | Augmentation added to (1) |
   | (3) | +Adaptive Adjacency ($\hat{\mathbf{A}} = \mathbf{A} + \mathbf{B} + \mathbf{C}$) | Adaptive topology added to (1) |
   | (4) | +Four-Stream Fusion | Multi-stream fusion applied on top of (3) |

**[TBD] flags:** None.

**Transition out:** "The evaluation framework rests on a set of testing decisions described in the following section."

---

## 2.7 Testing Strategy

**Target length:** Short–Medium (~200 words, 4 paragraphs)

**Purpose:** Centralise all testing methodology decisions in one place. Prevents testing considerations from appearing as afterthoughts in individual sections. Covers test set discipline, validation role, metric selection, reproducibility, and augmentation at inference.

**Paragraph structure:**

1. *Topic: Test set discipline.*
   The held-out Penn Action test set is evaluated exactly once per configuration, after training is fully finalised. This single-evaluation discipline prevents inadvertent test-set contamination — a standard requirement for unbiased reporting in supervised learning. Validation accuracy is never used to select between experimental configurations.

2. *Topic: Train/validation split and validation role.*
   The official Penn Action training partition is divided 80/20 (stratified by class) into training and validation subsets (~1,545 train / ~387 val). The validation set serves two purposes only: monitoring convergence during training and hyperparameter selection via the WandB sweep (§3.7). It is not consulted when comparing ablation conditions.

3. *Topic: Evaluation metrics.*
   Top-1 classification accuracy on the test set is the primary metric, consistent with prior Penn Action benchmarks [1, 2]. Per-class accuracy is additionally reported separately for the 8 exercise classes and 7 non-exercise classes, enabling direct assessment of whether improvements disproportionately benefit exercise recognition — the central research question (see §4.2).

4. *Topic: Reproducibility — seeds and augmentation at inference.*
   Each configuration is trained with 3 independent random seeds (seed values 42, 43, 44); mean ± std test accuracy is reported to assess stability. All random number generators — NumPy, PyTorch, and Python's \texttt{random} — are seeded identically. Augmentation transforms are applied exclusively during training; all evaluation uses original, unmodified sequences to ensure a fair and reproducible test condition.

**Citations:** `yan2018spatial` [1], `shi2019two` [2]

**[TBD] flags:**
- Confirm exact train/val/test counts from data loader logs (~1,545 / ~387 / ~394).
- Confirm seed values used for the 3 runs (42, 43, 44 assumed).

**Transition out:** "With the experimental design and testing framework established, the following sections describe the project management approach and version control strategy."

---

## 2.8 Project Management

**Target length:** Medium (~300 words across 2 subsections)

### 2.7.1 Agile Sprint Structure

**Purpose:** Show that the project was managed methodically, not ad-hoc. The sprint structure maps directly to the ablation conditions, demonstrating disciplined scope management.

**Paragraph structure:**

1. *Topic: Sprint methodology aligned to ablation conditions.*
   The project adopted a four-sprint Agile methodology, with each two-week sprint producing one working, validated experimental condition. This structure ensured that each ablation deliverable was independently completable and testable. Present the sprint table (see below).

**Sprint plan table:**

| Sprint | Duration | Goal | Key Deliverable |
|--------|----------|------|-----------------|
| Sprint 1 | Weeks 1–2 | Dataset pipeline and baseline model | Condition (1): baseline test accuracy established |
| Sprint 2 | Weeks 3–4 | Skeleton data augmentation | Condition (2): augmentation ablation result |
| Sprint 3 | Weeks 5–6 | Adaptive adjacency matrices | Condition (3): adaptive adjacency ablation result |
| Sprint 4 | Weeks 7–8 | Four-stream fusion and final evaluation | Condition (4): full model result; ablation table complete |

### 2.7.2 Risk Management

**Purpose:** Demonstrate awareness of the three main project risks and how each was pre-empted.

**Paragraph structure:**

1. *Topic: Two risks and mitigations.*
   - **Compute risk:** Colab Pro session limits mitigated by writing self-contained training scripts with checkpoint saving.
   - **Deadline risk:** The most compute-intensive sprint (Sprint 4, four-stream) was scoped last, ensuring a valid stopping point existed at every prior condition.

---

## 2.8 Version Control

**Target length:** Short (~200 words across 2 subsections)

### 2.8.1 Google Colab and GitHub Workflow

**Purpose:** Explain the two-phase version control strategy: exploratory (Colab-native) and canonicalised (GitHub).

**Paragraph structure:**

1. *Topic: Two-phase workflow.*
   During active experimentation, Colab's built-in revision history and Drive autosave preserved notebook state, enabling free cell-level iteration. At the end of each sprint, notebooks were canonicalised — non-essential cells removed, outputs cleared, execution order linearised — and committed to the project's GitHub repository. This produced a clean, reviewable snapshot at each ablation stage.

2. *Topic: Repository structure.*
   The repository is organised as: `data/` (preprocessing scripts), `models/` (baseline and adaptive ST-GCN definitions), `training/` (training loop and sweep config), `evaluation/` (test evaluation and confusion matrix scripts), and `notebooks/` (canonicalised sprint notebooks). See Appendix C.4 for the repository URL.

### 2.9.2 Branching Strategy

**Paragraph structure:**

1. *Topic: Branch-per-sprint and merge gating.*
   The `main` branch holds only canonicalised, sprint-end snapshots. Feature branches (`feature/augmentation`, `feature/adaptive-adj`, `feature/four-stream`) were created at the start of each sprint and merged via pull request after sprint acceptance criteria were met. The four-stream branch was merged only after all four stream models achieved convergence.

---

## Figures and Tables Needed in Chapter 2

| Item | Type | Section | Notes |
|------|------|---------|-------|
| Figure 2a — Baseline 6-Block ST-GCN Architecture | TikZ figure | §2.2 | Stack of 6 blocks with channel widths, temporal resolutions, strides, classification head. See diss-diagrams spec in §2.2. |
| Sprint Plan table | LaTeX table | §2.8.1 | 4 rows × 4 cols. Inline, not floating. |

---

## Cross-Reference Checklist

- §2.2 → cross-reference Figure `fig:baseline-arch` (Figure 2a, this chapter) and `yan2018spatial` [1]
- §2.3 → cross-reference `shorten2019survey` [9], `xin2024enhancing` [9]; forward-ref §3.3 for augmentation implementation
- §2.4 → cross-reference eq. `eq:adaptive-adj` (Chapter 1 §1.2.2) and `shi2019two` [2]; forward-ref §3.4 for implementation
- §2.5 → cross-reference `shi2020skeleton` [3]; forward-ref §3.5 for stream derivation; forward-ref §4.1 for per-stream analysis
- §2.6 → forward-reference §3.6 for hyperparameters; forward-reference Chapter 4 for results
- §2.7 → forward-ref §4.2 for per-class results; forward-ref §3.7 for WandB sweep
- §2.8 → cross-reference WandB [10] for sweep tracking as sprint log

---

## Transitions Summary

| From | To | Transition phrase |
|------|----|-------------------|
| Chapter intro | §2.1 | "The chapter begins by formally defining the recognition task…" |
| §2.1 | §2.2 | "The baseline architecture is designed to satisfy this objective with minimal architectural complexity." |
| §2.2 | §2.3 | "This baseline is extended through three sequential modifications, each targeting a different source of inaccuracy." |
| §2.3 | §2.4 | "The second modification targets the graph topology rather than the input data." |
| §2.4 | §2.5 | "The third modification extends the input representation from a single stream to four complementary modalities." |
| §2.5 | §2.6 | "With the three improvements defined, the following section specifies the ablation study design used to evaluate them." |
| §2.6 | §2.7 | "The evaluation framework rests on a set of testing decisions described in the following section." |
| §2.7 | §2.8 | "With the experimental design and testing framework established, the following sections describe the project management approach and version control strategy." |
| §2.9 | Chapter 3 | "The implementation of each design decision described in this chapter is detailed in Chapter~3." |

---

## Key Differences from v2 Brief (what changed)

| v2 (old) | v3 (new) |
|----------|----------|
| §2.1–2.4 Design, §2.2 Implementation, §2.3 Evaluation | §2.1 Problem Formulation, §2.2–2.5 Design rationale, §2.6 Ablation design, §2.7 Project Mgmt, §2.8 Version Control |
| Implementation details (data pipeline, WandB, training cycle) in Chapter 2 | Implementation details **moved to Chapter 3** |
| Evaluation metrics in Chapter 2 | Evaluation metrics **moved to Chapter 4** |
| 4 augmentations | **5 augmentations** (adds random scaling, σ=0.01 noise, rotation ±30°, crop 80–100%, speed 0.8–1.2×) |
| Fusion weights [TBD] | Fusion weights fixed: **2:1:2:1** from [3] |
| ~~SGD~~ (skeleton C.2) | **Adam** optimiser (confirmed by Chris — skeleton Appendix C.2 is incorrect) |
| 200 epochs with early stopping | **60 epochs** (no early stopping in final training) |
| 24-run grid search | **20-run Bayesian W&B sweep** with HyperBand early termination |
| 1163/1163 train/test (50/50) | **80/20 stratified split** of training partition → ~1,545 train, ~387 val, ~394 test |
| 13 classes (skeleton v3) | **15 classes** (confirmed by Chris — full Penn Action label set used) |
| No project management section | **§2.7 Project Management** (Agile sprints + risk management) |
| No version control section | **§2.8 Version Control** (Colab + GitHub + branching) |
| No Figure 2a | **Figure 2a** — Baseline 6-block ST-GCN architecture (mandatory) |

---

## Open Questions for Chris to Resolve Before Writing

1. Confirm the exact training/validation/test sample counts from your data loader logs (~1,545 / ~387 / ~394 — verify).
2. ~~13 vs 15 classes~~ — **resolved: 15 classes**.
3. ~~Optimiser~~ — **resolved: Adam** (skeleton Appendix C.2 says SGD but that is incorrect for your implementation).
4. ~~Rotation range~~ — **resolved: ±30°** (θ ~ U(−30°, 30°)).
5. GitHub repository URL for §2.8 / Appendix C.4.
6. GPU type for Appendix B.4 and §3.6.

