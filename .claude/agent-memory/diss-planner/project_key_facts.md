---
name: Dissertation Key Experimental Facts
description: Confirmed implementation facts for the ST-GCN dissertation — use these to override skeleton where they conflict
type: project
---

These facts are confirmed by Chris and CLAUDE.md and take precedence over the dissertation skeleton where they conflict.

**Architecture:** 6-block ST-GCN, channel widths [32, 32, 64, 64, 128, 128], Penn Action (13 joints, 15 classes).

**Optimiser: Adam** (NOT SGD — skeleton Appendix C.2 incorrectly states SGD; chapter2.tex line 28 and CLAUDE.md confirm Adam).

**Training:** 60 epochs, no early stopping, batch size 64, lr=0.001, cosine annealing (T_max=60, eta_min=1e-6), weight decay 1e-5, dropout 0.2.

**Hyperparameter search:** 20-run Bayesian W&B sweep on baseline only, HyperBand early termination (min_iter=10, eta=3), runs capped at 20 epochs.

**Data split:** 80/20 stratified split of official training partition → ~1,545 train / ~387 val / ~394 test (verify exact counts).

**Seeds:** 42, 43, 44 (3 runs per condition). NumPy, PyTorch, and Python random all seeded identically.

**Augmentations (5, training only):** random rotation ±30°, random scaling (0.9–1.1×), Gaussian joint jitter (σ=0.01), random temporal crop (80–100%), speed perturbation (0.8–1.2×). NOTE: skeleton §3.3 says ±15° rotation but CLAUDE.md says ±30° — Chris must confirm from code.

**Adaptive adjacency:** 2S-AGCN decomposition Â = A + B + C per block. B is nn.Parameter(13,13) init to zero. C via θ/φ 1×1 conv branches, row-wise softmax. Einsum: 'nctv,nvw->nctw'. Â must be row-normalised before GCNConv (bug found and fixed).

**Four-stream fusion:** joint, bone, joint-motion, bone-motion. Score-level weighted average 2:1:2:1 from MS-AAGCN [3]. Models trained independently.

**Known bugs found and fixed (document in Chapter 3):**
1. Augmentation applied with training=False on training dataset — caught and corrected.
2. Row-normalisation missing from Â — caused BatchNorm destabilisation, corrected.

**Key results:**
- Augmentation-only: ~83.80% (best single-stream, positive finding).
- Adaptive adjacency: ~79–80% (negative finding, attributed to overfitting on small dataset).
- Four-stream fusion: in progress.

**Compute:** Google Colab Pro. GPU type TBD (Chris to confirm from session info).

**Why:** These facts govern all writing decisions. The Adam/SGD discrepancy in particular must be flagged every time §3.6 or Appendix C.2 is discussed.

**How to apply:** Always use these values when writing any section. Flag [TBD] for anything marked uncertain above.
