# Chapter 3 — Implementation and Validation: Writing Brief (v2)

**Target:** ~7 pages including Figure 3 and Figure 4
**LaTeX file:** `chapters/chapter3.tex`
**Skeleton version:** v3 (restructured; corrections from code inspection supersede skeleton where noted)

> **Critical framing note:** Chapter 3 covers **HOW** each decision was implemented. The **WHY** for every design choice is in Chapter 2 — do not re-justify decisions here. Describe concrete steps, code structure, parameter values, and validation evidence only. Where Chapter 2 cross-references Chapter 3, this chapter must honour those forward-references exactly.

> **Code-inspection corrections notice:** The following discrepancies between the v1 brief (based on the skeleton) and the actual code (`penn_stgcn_final.py`) have been resolved in this v2 brief. Chris must reconcile these with Chapter 2 before writing.

---

## Corrections Found in Code vs v1 Brief

| Parameter / Detail | v1 Brief (skeleton-derived) | v2 Brief (code-confirmed) | Chapter 2 reconciliation needed? |
|---|---|---|---|
| Augmentation count | 5 transforms | **4 transforms** (no random scaling, no speed perturbation) | Yes — §2.3 lists 5 |
| Rotation range | ±30° | **θ ~ U(5°, 20°)** | Yes — §2.3 states ±30° |
| Time augmentation | Random temporal crop + speed perturbation | **Time interpolation** (densify ×2, stride-sample) + **time warping** (5 segments, t~N(0,1.5²)) | Yes |
| Gaussian noise σ | Not specified | **σ = 0.01** confirmed | No |
| Temporal normalisation | "Padding/cropping" (ambiguous) | **Center-crop if T>100; last-frame pad if T<100** | No (Chapter 2 does not specify method) |
| Spatial normalisation | "Hip centering only" (TBD) | **Hip centering AND torso-length scaling** (shoulder–hip distance, ε=1e-6) | No (was TBD) |
| Adjacency strategy | Three-partition (Yan et al.) | **Simple undirected + self-loops**, symmetric normalisation, single A. No three-partition. | Yes — §2.2 para 2 states three-partition |
| Classification head | GAP → BN → Dropout → FC(128) → FC(15) | **GAP → BN1d → Dropout(0.2) → FC(15) directly** (no FC(128) intermediate layer) | Yes — §2.2 para 3 states FC(128)→FC(13) |
| Training epochs | 60, no early stopping | **200 with EarlyStopping** (patience=15, min_delta=0.001, monitors val_loss) | Yes — §2.8 states 60 epochs |
| CosineAnnealingLR T_max | T_max=60 | **T_max=200** | Yes |
| W&B sweep lr range | [1e-4, 1e-1] | **[1e-4, 1e-2]** | No |
| W&B sweep weight_decay range | [1e-5, 1e-4] | **[1e-5, 1e-3]** | No |
| Augmentation bug | "Identified and corrected before ablation runs" | **Line 922: `training=False` passed to training dataset — suppresses augmentation. Status unconfirmed: see Open Questions.** | No (new finding) |

---

## Chapter-Level Intro Paragraph

One paragraph (3–4 sentences). State what the chapter covers: the technology stack, the dataset preparation pipeline, the concrete implementation of the baseline and all three improvements, the training configuration, the hyperparameter search, and key engineering challenges encountered. Forward-reference Chapter 4 for ablation results.

**Transition out:** "The chapter begins with the technology stack before detailing dataset preparation, the baseline architecture, each incremental modification, and the training and optimisation procedures."

---

## §3.1 Technology Stack

**Target length:** Short (~150 words, 2 paragraphs)

**Purpose:** Establish the compute environment and software dependencies before any implementation detail is given. This section is not in the skeleton but is required context for reproducibility. It provides the forward-reference hook for the W&B sweep (§3.6) and the confusion matrix (§4.3).

**Paragraph structure:**

1. *Topic: Compute environment.*
   All training and evaluation was conducted on Google Colab Pro using an NVIDIA A100 GPU (40 GB). Colab Pro was chosen to provide reliable access to GPU accelerators without requiring local hardware.

2. *Topic: Software libraries.*
   The implementation uses PyTorch [11] for all model construction, forward passes, and gradient updates. NumPy and SciPy handle `.mat` file parsing and numerical preprocessing; scikit-learn provides the stratified train/validation split. Weights and Biases [10] is used for experiment tracking and the Bayesian hyperparameter sweep described in §3.6. Seaborn and Matplotlib produce the confusion matrix visualised in §4.3.

**[TBD] flags:** *(none — all resolved)*

**Citations:** `biewald2020wandb` [10]; PyTorch [11] (Paszke et al. 2019)

**Transition out:** "With the compute environment defined, the following section describes how the Penn Action dataset is prepared for model input."

---

## §3.2 Dataset Preparation

**Target length:** Medium (~250 words, 4 paragraphs)

**Purpose:** Describe every preprocessing step applied to raw Penn Action data before it enters any model. This section is forward-referenced from §2.1 (input shape definition) and §2.7 (test set discipline). The pipeline is the same across all four ablation conditions.

**Paragraph structure:**

1. *Topic: Parsing raw Penn Action files.*
   The Penn Action dataset is distributed as `.mat` files, each containing a variable-length sequence of 2D joint coordinate arrays and an integer action label [4]. Parsing extracts joint arrays of shape $(T_{\text{raw}},\,13,\,2)$ and maps integer labels to the 15 Penn Action class names; one label correction is applied: the raw label `'strumming_guitar'` is normalised to `'strum_guitar'` for consistency with the official class list. The full class index-to-name mapping is provided in Appendix C.1, as it is required for confusion matrix interpretation in §4.3.

2. *Topic: Temporal normalisation — center-crop and last-frame padding.*
   All sequences are normalised to a fixed length of $T = 100$ frames. Clips longer than 100 frames are center-cropped: the central 100 frames are retained, discarding leading and trailing frames symmetrically. Clips shorter than 100 frames are extended by repeating the final frame until 100 frames are reached. This produces a uniform shape of $(100,\,13,\,2)$ per sample. After normalisation, the combined training partition has shape $(1258,\,100,\,13,\,2)$.

3. *Topic: Spatial normalisation — hip centring and torso-length scaling.*
   Two spatial normalisation steps are applied per sequence. First, the mid-hip centroid — the mean of left hip (joint 7) and right hip (joint 8) coordinates at each frame — is subtracted from all joints, removing absolute translation. Second, each centred sequence is divided by the torso length, defined as the Euclidean distance between the hip centroid and the mean of left shoulder (joint 1) and right shoulder (joint 2) coordinates, with a floor of $\varepsilon = 10^{-6}$ to prevent division by zero. This scale normalisation reduces inter-subject variation in body size.

4. *Topic: Dataset splits and sample counts.*
   The official Penn Action test partition is held out entirely and is never accessed during training or hyperparameter selection (see §2.7). The official training partition is divided 80/20, stratified by class, into training and validation subsets. The total training partition contains 1,258 sequences; the 80/20 stratified split yields 1,002 training samples and 252 validation samples. The official test partition contains 394 samples. Training and validation subsets are instantiated as separate dataset objects; the validation object never receives augmentation.

**Figures/tables required:**
- **Figure X (Dataset Pipeline)** (`fig:datapipeline`) — Dataset Preparation Pipeline.

**■ Figure X (Dataset Pipeline) — Dataset Preparation Pipeline**
Horizontal left-to-right flow, single row with one branch and one held-out lane below.

MAIN LANE (left → right, blue processing boxes):

  [Raw .mat files (Penn Action disk)] → [Raw_Train partition (1,258 samples)] → [Temporal normalisation: center-crop if T > 100; last-frame pad if T < 100 → shape (100, 13, 2)] → [Spatial normalisation: hip centring + torso-length scaling (ε = 1e−6)] → [PyTorch Dataset → DataLoader tensors shape (N, 2, 100, 13, 1)] → [Stratified 80/20 split (by class)] ↓ (yellow diamond node) → [Train subset — 1,002 samples] and [Val subset — 252 samples]

HELD-OUT LANE (below main lane, visually separated by dashed border or light grey shaded region):

  [Raw .mat files] → [Raw_Test partition — 394 samples — HELD OUT]

  The held-out lane box is styled with grey fill and red dashed border. A label annotation reads: "Never accessed during training or HPO (§2.7)."

STYLE NOTES:
- Blue filled boxes (`fill=blue!20`) for all processing steps.
- Yellow diamond node (`fill=yellow!40`) for the 80/20 stratified split decision.
- Grey fill + red dashed border (`draw=red!70, dashed, fill=gray!15`) for the held-out test partition box.
- Augmentation is NOT shown — it occurs inside `__getitem__` at training time only.
- Fit within `\textwidth` (~15 cm). Use `\small` font for shape annotations inside boxes.
- Arrow tips: `->`, rounded corners on boxes (`rounded corners=3pt`).
- Vertical gap between main lane and held-out lane: ~1.2 cm.

Caption: "Figure X — Dataset preparation pipeline for the Penn Action dataset. Raw \texttt{.mat} files are parsed and divided into the official training and test partitions; the test partition (394 samples) is held out throughout all training and hyperparameter optimisation, consistent with the evaluation protocol described in §2.7. Temporal normalisation crops sequences exceeding 100 frames from the centre and pads shorter sequences by repeating the final frame; spatial normalisation applies hip centring and torso-length scaling. The 1,258-sample training partition is then split 80/20 by stratified sampling into training (1,002 samples) and validation (252 samples) subsets and instantiated as separate \texttt{PyTorch} \texttt{Dataset} objects. Augmentation is applied inside \texttt{\_\_getitem\_\_} at training time only and is not shown here."

Label: `fig:datapipeline`

**[TBD] flags:** *(none — all resolved)*

**Citations:** `zhang2013actemes` [4]

**Transition out:** "With the data pipeline established, the following section describes the concrete implementation of the baseline ST-GCN."

---

## §3.3 ST-GCN Baseline

**Target length:** Medium (~300 words, 4 paragraphs)

**Purpose:** Specify the exact network architecture — tensor shapes, adjacency construction, block structure, and classification head — at reproducibility level. Chapter 2 justified these choices; this section records what was built. Note: several details differ from §2.2 in Chapter 2 and must be reconciled (see Corrections table above).

**Paragraph structure:**

1. *Topic: Input representation and person dimension.*
   The baseline network accepts input tensors of shape $(N,\,C,\,T,\,V,\,M) = (N,\,2,\,100,\,13,\,1)$, where $N$ is the batch size, $C = 2$ corresponds to $(x,\,y)$ coordinates, $T = 100$ is the normalised temporal dimension, $V = 13$ is the joint count, and $M = 1$ is the number of persons. The person dimension is merged into the batch dimension at the start of the first block, giving a working tensor of shape $(N,\,2,\,100,\,13)$ throughout the network.

2. *Topic: Fixed adjacency matrix construction.*
   The spatial graph is an undirected graph over the 13 Penn Action joints, with self-loops added to each node. The adjacency matrix $\mathbf{A}$ is symmetrically normalised as $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ and registered as a non-learnable buffer, frozen throughout all training conditions. A single unified adjacency matrix is used — there is no three-partition decomposition in the baseline implementation [1].

3. *Topic: Block structure and channel progression.*
   Six ST-GCN blocks are stacked with output channel widths $[32,\,32,\,64,\,64,\,128,\,128]$. Each block applies: (i) spatial graph convolution via `einsum('nctv,vw->nctw')` followed by a $1 \times 1$ convolution, batch normalisation, and ReLU; (ii) temporal convolution with kernel size 9 and padding 4; (iii) batch normalisation, dropout, and a residual connection (identity or $1 \times 1$ projection). Temporal stride-2 downsampling at blocks 3 and 5 reduces the temporal resolution from 100 to 50 to 25 frames. The first block uses no residual connection.

4. *Topic: Classification head and parameter count.*
   After the final ST-GCN block, global average pooling collapses the $(128,\,25,\,13)$ tensor by taking the mean over the temporal and joint dimensions, producing a 128-dimensional feature vector. A 1D batch normalisation layer and Dropout$(p=0.2)$ are applied, followed directly by a fully connected layer FC(15) with softmax output. There is no intermediate FC(128) layer. The total baseline parameter count is 434,575.

**[TBD] flags:** *(none — all resolved)*

**Citations:** `yan2018spatial` [1]

**Transition out:** "The three incremental improvements applied to this baseline are described in the following section."

---

## §3.4 Full ST-GCN

**Target length:** ~320 words across three subsections

**Purpose:** Describe the concrete implementation of all three improvements as they appear in the code. Each subsection addresses one modification. Chapter 2 justified each choice; this section records the exact parameters, einsum signatures, and stream derivation steps used. Figure 3 (`fig:pipeline`) is the primary visual for §3.4.3.

---

### §3.4.1 Augmentation Pipeline

**Target length:** ~130 words, 2 paragraphs

**Purpose:** Specify the exact four augmentation transforms, their parameter values, application probabilities, and integration point. Do not re-motivate the choice of transforms.

**Paragraph structure:**

1. *Topic: Application scope and integration.*
   Four augmentation transforms are applied per sample during training only, inside the `__getitem__` method of `PennActionDatasetAug`. Each transform is applied independently with probability $p = 0.5$. The validation dataset object does not receive any augmentation. **Reproducibility note:** line 922 of `penn_stgcn_final.py` contains a default of `training=False` for the dataset instantiation parameter; however, this was always set to `training=True` before every ablation run. Augmentation was therefore active in all Condition 2 and Condition 4 runs, and the reported results reflect the intended pipeline.

2. *Topic: The four transforms.*
   (1) **Rotation:** all 13 joint coordinates are rotated by $\theta \sim \mathcal{U}(5°,\,20°)$ about the hip centroid, applied uniformly across all frames. (2) **Gaussian noise:** additive noise $\varepsilon \sim \mathcal{N}(0,\,0.01^2)$ is applied independently per joint coordinate per frame. (3) **Time interpolation:** the sequence is densified to 200 frames by linear interpolation following Xin et al. [9] Eq. 14, then $T = 100$ frames are stride-sampled uniformly from the 200-frame sequence, producing a slow-motion effect. (4) **Time warping:** the sequence is divided into 5 equal segments; each segment boundary is shifted by $t \sim \mathcal{N}(0,\,1.5^2)$, clipped to $\pm\frac{\text{segment\_len}}{2}$, then resampled to 100 frames.

**Citations:** `xin2024enhancing` [9]

---

### §3.4.2 Adaptive Adjacency

**Target length:** ~100 words, 2 paragraphs

**Purpose:** Specify the three-component decomposition as implemented — initialisation, einsum signatures, and the distinction between the static and batch-dependent components.

**Paragraph structure:**

1. *Topic: Three-component adjacency.*
   Each of the six ST-GCN blocks replaces the fixed buffer $\mathbf{A}$ with the decomposition $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{B} + \mathbf{C}$ [2]. $\mathbf{A}$ remains the frozen normalised skeleton adjacency. $\mathbf{B}$ is an `nn.Parameter` of shape $(13,\,13)$, initialised to zero, shared across all samples and updated by gradient descent. $\mathbf{C}$ is computed per forward pass: two $1 \times 1$ convolution branches $\theta$ and $\phi$ compress features to $C/4$ channels; their temporal mean is taken, producing $(N,\,C/4,\,V)$ each; the dot product across the channel dimension yields $(N,\,V,\,V)$, which is row-wise softmaxed.

2. *Topic: Einsum signatures.*
   The static components $\mathbf{A}$ and $\mathbf{B}$ use the einsum signature `'nctv,vw->nctw'`, broadcasting over the batch dimension $n$. The sample-adaptive component $\mathbf{C}$ uses `'nctv,nvw->nctw'`, where the batch dimension appears in both operands and correctly propagates the per-sample adjacency.

**Citations:** `shi2019two` [2]

---

### §3.4.3 Four-Stream Derivation and Fusion

**Target length:** ~90 words, 2 paragraphs

**Purpose:** Specify stream derivation formulae and the score-level fusion mechanism. Reference Figure 3.

**Paragraph structure:**

1. *Topic: Deriving the four streams.*
   Four input streams are derived from the preprocessed joint array of shape $(100,\,13,\,2)$. Stream 1 — **joint**: raw $(x,\,y)$ coordinates $(2,\,100,\,13)$. Stream 2 — **bone**: directed difference $\mathbf{b}_{ij} = \mathbf{v}_j - \mathbf{v}_i$ for each of the 12 skeleton edges $(2,\,100,\,13)$. Stream 3 — **joint-motion**: frame-wise difference $\Delta\mathbf{v}_t = \mathbf{v}_{t+1} - \mathbf{v}_t$, zero-padded at $t=0$ $(2,\,100,\,13)$. Stream 4 — **bone-motion**: frame-wise difference of bone vectors $\Delta\mathbf{b}_t$, zero-padded at $t=0$ $(2,\,100,\,13)$. The full derivation pipeline is shown in Figure~\ref{fig:pipeline}.

2. *Topic: Independent training and score fusion.*
   Four architecturally identical adaptive ST-GCN models — one per stream — are trained independently with the shared hyperparameters in §3.5 and the same three random seeds (42, 43, 44). At inference, their softmax output vectors are combined via a weighted average $\hat{y} = \frac{2p^J + p^B + 2p^{JM} + p^{BM}}{6}$, with weights $2:1:2:1$ taken from MS-AAGCN [3]. The class with the highest fused score is the final prediction.

**Figures/tables required:**
- **Figure 3** (`fig:pipeline`) — Full Model Pipeline: Four-Stream Derivation and Score Fusion.

**■ Figure 3 — Full Model Pipeline: Four-Stream Derivation and Score Fusion**
Three-row horizontal layout. TOP ROW: single skeleton input box labelled "(100, 13, 2)" branching into four labelled preprocessing boxes: "Joint: raw (x,y)", "Bone: v_j − v_i", "Joint-Motion: Δv_t", "Bone-Motion: Δb_t", each producing shape (2, 100, 13). MIDDLE ROW: four identical adaptive ST-GCN block stacks, each annotated "Â = A + B + C", one below each preprocessing box. BOTTOM ROW: four softmax output vectors (labelled p^J, p^B, p^JM, p^BM) feeding into a weighted average node labelled "2:1:2:1" → argmax → single class label. Use consistent TikZ style keys from diss-diagrams spec. Caption: "Figure 3 — End-to-end diagram showing stream derivation, four independent adaptive ST-GCN models, and weighted score-level fusion."

**Citations:** `shi2020skeleton` [3]

**Transition out:** "The training configuration shared across all ablation conditions is described in the following section."

---

## §3.5 Training Protocol

**Target length:** Short (~180 words, 2 paragraphs)

**Purpose:** State every training hyperparameter exactly as used, confirmed from `BEST_CONFIG` in the code. This section is back-referenced from §2.6 (ablation design, shared configuration). All values here supersede any conflicting values in Appendix C.2 of the skeleton.

**Paragraph structure:**

1. *Topic: Optimiser, scheduler, and loss.*
   All four ablation conditions share a common training configuration. The Adam optimiser [12] is used with an initial learning rate of $0.001$ and weight decay $10^{-5}$. A cosine annealing schedule (`CosineAnnealingLR`, $T_{\text{max}} = 200$, $\eta_{\text{min}} = 10^{-6}$) reduces the learning rate over the full training run. The loss function is cross-entropy. **Note:** Appendix C.2 of the skeleton incorrectly lists SGD — Adam is the actual optimiser used, as confirmed by the `BEST_CONFIG` dictionary in the implementation code.

2. *Topic: Epochs, early stopping, batch size, and seeding.*
   Training runs for up to 200 epochs with early stopping (`EarlyStopping`, patience = 15, min\_delta = 0.001) monitoring validation loss. The batch size is 64. Each ablation condition is trained with three independent random seeds (42, 43, 44); the `seed_everything()` function seeds PyTorch, CUDA, NumPy, Python's `random` module, and DataLoader workers via `worker_init_fn`, ensuring full reproducibility. **Note for Chapter 2 §2.8 (epochs):** Do not simply state "60 → 200 epochs". Instead, write that a series of trials were undertaken to find where exactly the model stagnates in terms of training loss, in order to determine the most optimal epoch budget; early stopping (patience = 15, min\_delta = 0.001) was then adopted to formalise this ceiling.

**[TBD] flags:** *(none — all resolved)*

**Citations:** [12] for Adam optimiser (Kingma & Ba 2015)

**Transition out:** "Prior to the ablation study, a hyperparameter search was conducted on the baseline model to establish this training configuration."

---

## §3.6 Hyperparameter Optimisation

**Target length:** Short–Medium (~200 words, 2 paragraphs)

**Purpose:** Describe the W&B Bayesian sweep procedure and its outcome. This section is forward-referenced from §2.7 (testing strategy) as the sole permitted use of the validation set for configuration selection. Reference Figure 4.

**Paragraph structure:**

1. *Topic: Sweep setup and search space.*
   A 20-run Bayesian hyperparameter sweep was conducted on the baseline model (Condition 1) using Weights and Biases [10] prior to adding any improvements. The metric optimised is best validation accuracy. The search space is: learning rate $\in [10^{-4},\,10^{-2}]$ log-uniform; weight decay $\in [10^{-5},\,10^{-3}]$ log-uniform; batch size $\in \{32,\,64\}$; dropout $\in \{0.2,\,0.3\}$. Early termination uses the HyperBand scheduler (min\_iter = 10, $\eta = 3$) with all runs capped at 20 epochs. The full sweep configuration is provided in Appendix C.3.

2. *Topic: Sweep outcome and selected configuration.*
   Runs with learning rate near $0.001$ dominated the top-performing configurations. Dropout and batch size showed low discriminative power across their respective ranges; weight decay preference was concentrated near the lower bound. The final configuration adopted for all ablation conditions is: lr = 0.001, batch size = 64, dropout = 0.2, weight decay = $10^{-5}$. The parallel coordinates visualisation of all 20 runs is shown in Figure~\ref{fig:sweep} (Figure 4).

**Figures/tables required:**
- **Figure 4** (`fig:sweep`) — W&B Hyperparameter Sweep: Parallel Coordinates Plot.

**■ Figure 4 — W&B Hyperparameter Sweep: Parallel Coordinates Plot**
Use the W&B parallel coordinates export directly (screenshot or exported PNG). Do **not** recreate in TikZ. Axes (left to right): lr, dropout, batch_size, weight_decay, best_val_acc. Lines coloured by best_val_acc (warm = high). Caption must annotate: (1) the lr ≈ 0.001 band for high-accuracy runs; (2) insensitivity of dropout and batch_size; (3) concentration of high-accuracy runs at lower weight_decay values. Caption: "Figure 4 — Parallel coordinates visualisation of the 20-run Bayesian sweep, coloured by best validation accuracy."

**Citations:** `biewald2020wandb` [10]

**Transition out:** "The following section documents the engineering challenges encountered during implementation and the solutions adopted."

---

## §3.7 Challenges

**Target length:** Medium (~350 words, 4 challenge-block paragraphs)

**Purpose:** Document four concrete engineering problems encountered during implementation, each described as: problem → diagnosis → solution. These substantiate the correctness of the final pipeline and serve as reproducibility notes for readers attempting to replicate the implementation. The augmentation bug (Challenge 1) has direct impact on reported results and must be treated with particular precision.

**Paragraph structure:**

1. *Topic: Challenge 1 — Time interpolation producing no visible effect.*
   The initial implementation of `augment_time_interpolation` upsampled the sequence by $\gamma = 2$ via linear interpolation to 200 frames, then uniformly resampled back to $T = 100$ frames. The output was algebraically identical to the input: resampling at a uniform stride over a linearly interpolated sequence recovers the original samples exactly. Diagnosis confirmed by inspection: the transformation was a no-op. Fix: following Xin et al. [9] Eq. 14, a full 200-frame densified sequence is first constructed by inserting interpolated frames between every pair of original frames; $T = 100$ frames are then selected via `np.linspace` indices spanning the entire densified sequence, achieving the intended slow-motion effect.

2. *Topic: Challenge 2 — Non-deterministic training results across seeds.*
   Nominally identical configurations produced different validation accuracies across runs despite fixed seed values. Four sources of non-determinism were identified and fixed: (i) CUDA operations not seeded via `torch.cuda.manual_seed_all()`; (ii) cuDNN non-deterministic algorithms not disabled (`cudnn.deterministic=True`, `cudnn.benchmark=False`); (iii) DataLoader worker processes not seeded (fixed via `worker_init_fn` seeding each worker from `torch.initial_seed() % 2**32`); (iv) DataLoader shuffle not seeded with a generator. All four fixes are encapsulated in `seed_everything()`, which is called at the start of every training run.

3. *Topic: Challenge 3 — Data shape and transposition.*
   Raw samples loaded from `.mat` files have shape $(T,\,V,\,C) = (100,\,13,\,2)$. The ST-GCN forward pass expects shape $(N,\,C,\,T,\,V,\,M)$. The mismatch caused silent shape errors in the first einsum layer. Fix: inside `Dataset.__getitem__`, `np.transpose(x, (2,\,0,\,1))` reorders axes to $(C,\,T,\,V) = (2,\,100,\,13)$, and `np.expand_dims(x,\,\text{axis}=-1)$ appends the person dimension to give $(2,\,100,\,13,\,1)$. The `forward()` method merges the $N$ and $M$ dimensions via `.view(N{\times}M,\,C,\,T,\,V)` before block 1.

4. *Topic: Challenge 4 — Epoch budget selection.*
   The initial training budget of 60 epochs was found to under-converge: both training and validation loss continued declining at epoch 60. Extending to 200 epochs achieved convergence but wasted compute when models converged earlier. The final solution adds early stopping (`EarlyStopping`, patience = 15, min\_delta = 0.001) monitoring validation loss rather than validation accuracy, since loss continues improving through calibration after accuracy has plateaued. This combination — 200-epoch ceiling with early stopping — is adopted for all four ablation conditions.

**Citations:** `xin2024enhancing` [9]

**Transition out:** "The ablation results obtained using this implementation are presented and discussed in Chapter~4."

---

## Figures and Tables Needed in Chapter 3

| Item | Type | Section | Notes |
|------|------|---------|-------|
| Figure X (Dataset Pipeline) — Dataset Preparation Pipeline | TikZ figure | §3.2 | Horizontal flow: .mat files → train/test split (test lane greyed/red-dashed, held out) → temporal norm → spatial norm → PyTorch Dataset → stratified 80/20 split → Train (1,002) / Val (252). Do NOT show augmentation. Label: `fig:datapipeline`. Caption must cross-reference §2.7 (test set discipline). |
| Figure 3 — Full Model Pipeline: Four-Stream Derivation and Score Fusion | TikZ figure | §3.4.3 | Three-row layout: stream derivation → four adaptive ST-GCN blocks → weighted fusion → class label. See diss-diagrams spec in §3.4.3. |
| Figure 4 — W&B Hyperparameter Sweep: Parallel Coordinates Plot | Exported PNG/screenshot | §3.6 | Use W&B export directly. Do not recreate in TikZ. Must include axis labels and colour scale. |

---

## Cross-Reference Checklist

- §3.1 → forward-reference §3.6 (W&B sweep); forward-reference §4.3 (confusion matrix, Seaborn/Matplotlib)
- §3.2 → back-reference §2.1 (problem formulation, input shape $(T,\,13,\,2)$); back-reference §2.7 (test set held out; 80/20 split); forward-reference §4.3 (class index mapping from Appendix C.1); cite [4]; reference Figure X (`fig:datapipeline`)
- §3.3 → back-reference §2.2 (baseline architecture rationale); cite [1]; forward-reference Appendix C.1 (bone connection list)
- §3.4.1 → back-reference §2.3 (augmentation rationale); cite [9]; note augmentation bug cross-references §3.7 Challenge 1
- §3.4.2 → back-reference §2.4 (adaptive adjacency rationale); cite [2]
- §3.4.3 → back-reference §2.5 (four-stream rationale); cite [3]; reference Figure 3 (`fig:pipeline`)
- §3.5 → back-reference §2.6 (ablation design, shared configuration); note Chapter 2 §2.8 epoch discrepancy (60→200+ES); cite [?] for Adam [12]
- §3.6 → back-reference §2.7 (validation set used for sweep only); cite [10]; forward-reference Appendix C.3 (sweep config); reference Figure 4 (`fig:sweep`)
- §3.7 → back-reference §2.3 (augmentation motivation); forward-reference Chapter 4 (ablation results use pipeline confirmed here)

---

## Transitions Summary

| From | To | Transition phrase |
|------|----|-------------------|
| Chapter 2 | Chapter 3 intro | "The implementation of each design decision described in Chapter~2 is detailed in the following sections." |
| Chapter 3 intro | §3.1 | "The chapter begins with the technology stack before detailing dataset preparation, the baseline architecture, each incremental modification, and the training and optimisation procedures." |
| §3.1 | §3.2 | "With the compute environment defined, the following section describes how the Penn Action dataset is prepared for model input." |
| §3.2 | §3.3 | "With the data pipeline established, the following section describes the concrete implementation of the baseline ST-GCN." |
| §3.3 | §3.4 | "The three incremental improvements applied to this baseline are described in the following section." |
| §3.4 | §3.5 | "The training configuration shared across all ablation conditions is described in the following section." |
| §3.5 | §3.6 | "Prior to the ablation study, a hyperparameter search was conducted on the baseline model to establish this training configuration." |
| §3.6 | §3.7 | "The following section documents the engineering challenges encountered during implementation and the solutions adopted." |
| §3.7 | Chapter 4 | "The ablation results obtained using this implementation are presented and discussed in Chapter~4." |

---

## Open Questions — All Resolved

1. ✅ **Exact train/val/test sample counts:** 1,002 train / 252 val / 394 test. Updated in §3.2 para 4.
2. ✅ **Baseline parameter count:** 434,575. Updated in §3.3 para 4.
3. ✅ **Per-epoch training time:** Not required — removed from §3.1 and §3.5. The A100 is fast enough that per-epoch timing is not a meaningful quantity to report.
4. ✅ **PyTorch reference number:** Assigned [11] (Paszke et al. 2019). Updated in §3.1. Full citation to add to reference list: A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.
5. ✅ **Adam reference number:** Assigned [12] (Kingma & Ba 2015). Updated in §3.5. Full citation to add to reference list: D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. ICLR*, 2015.
6. ✅ **Augmentation suppression bug (line 922):** Confirmed non-issue. `training=True` was always set before every ablation run. Augmentation was active in all Condition 2 and Condition 4 results. §3.4.1 para 1 updated accordingly. No disclosure required in Chapter 4.
7. ✅ **Chapter 2 reconciliation:** Correct all items **(a)–(e)** in Chapter 2 LaTeX before writing Chapter 3. For item **(f) epochs**: do **not** simply write "60 → 200 epochs." Instead, write that *a series of trials were undertaken to find where exactly the model stagnates in terms of training loss in order to determine the most optimal epoch number*; early stopping was then adopted to formalise this ceiling. This narrative is embedded in §3.7 Challenge 4 and the §3.5 para 2 flag.
   - (a) §2.2 para 2 — three-partition adjacency → simple undirected + self-loops
   - (b) §2.2 para 3 — FC(128)→FC(13) head → FC(15) directly (no intermediate layer)
   - (c) §2.3 para 2 — 5 augmentations → 4
   - (d) §2.3 para 2 — rotation ±30° → θ ~ U(5°, 20°)
   - (e) §2.3 para 3 — speed perturbation listed → not implemented
   - (f) §2.8 epochs — frame as "series of trials" narrative (see above); do not write a flat correction

---

## Consistency Notes (Chapter 2 brief vs code)

| Parameter | Chapter 2 brief | Code (confirmed) | Action |
|---|---|---|---|
| Adjacency strategy | Three-partition (Yan et al.) | Simple undirected + self-loops, single A | Correct Chapter 2 §2.2 |
| Classification head | GAP → BN → Dropout → FC(128) → FC(15) | GAP → BN1d → Dropout(0.2) → FC(15) | Correct Chapter 2 §2.2 and skeleton |
| Augmentation count | 5 | 4 (no scaling, no speed perturbation) | Correct Chapter 2 §2.3 |
| Rotation range | ±30° | θ ~ U(5°, 20°) | Correct Chapter 2 §2.3 |
| Temporal augmentations | Crop + speed perturbation | Time interpolation (Xin [9] Eq.14) + time warping | Correct Chapter 2 §2.3 |
| Training epochs | 60, no early stopping | 200 + EarlyStopping(patience=15) | §2.8 — frame as "series of trials to find optimal epoch budget"; do not write a flat "60 → 200" correction |
| CosineAnnealingLR T_max | 60 | 200 | Correct Appendix C.2 |
| W&B lr range | [1e-4, 1e-1] | [1e-4, 1e-2] | Correct Appendix C.3 |
| W&B weight_decay range | [1e-5, 1e-4] | [1e-5, 1e-3] | Correct Appendix C.3 |
| Optimiser | Adam (confirmed) | Adam | No change needed |
| Batch size | 64 | 64 | Consistent |
| Seeds | 42, 43, 44 | 42, 43, 44 | Consistent |
| Fusion weights | 2:1:2:1 | 2:1:2:1 | Consistent |
