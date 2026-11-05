---
name: Watch Authenticity CV Training
description: Train a calibrated, explainable fake/genuine classifier for luxury watches (Rolex Submariner MVP) from a frozen DINOv2 backbone, scraped marketplace photos, on AWS GPU infra.
topics: dinov2, vision-transformer, linear-probe, temperature-scaling, calibration, grad-cam, explainability, perceptual-hash, laplacian-blur-detection, web-scraping-legality, aws-ec2-gpu, counterfeit-detection
created: 2026-11-03
updated: 2026-11-03
scratchpad: .specs/scratchpad/67f8e09e.md
---

# Watch Authenticity CV Training

## Overview

Covers the full technical stack for training a per-angle, multi-class (genuine + fake grades 1-4 + rejection class) classifier on top of a frozen DINOv2 ViT-B/14 backbone: calibration, ViT-compatible explainability, dataset-quality gating, and training infra. Also covers the legal exposure of the planned scrape-labeled dataset — the single highest-risk item found in research is **Reddit**, not the marketplaces.

---

## Key Concepts

- **Linear probe**: train only a small classification head on frozen backbone features; unfreeze backbone layers only if accuracy saturates (task's own plan — matches DINOv2's published benchmark approach).
- **Temperature scaling**: single-scalar post-hoc calibration fit on held-out validation logits; fixes in-distribution miscalibration only, degrades under distribution shift.
- **Distribution shift (superclone risk)**: a model calibrated on known factories will still be miscalibrated on an unseen factory/grade-4 superclone — this is why the task mandates cross-factory holdout evaluation, not just a train/val split.
- **ViT-compatible Grad-CAM**: plain Grad-CAM targets CNN conv layers; on a Transformer you target a late block's norm layer instead (e.g. `model.blocks[-1].norm1`).
- **Source-vs-subject confound**: fakes disproportionately come from forum/OLX-style photos, genuines from Chrono24/dealer-style photos — the model can learn "photo style" instead of "watch." Mitigate with tight crops, photographic-style augmentation, and cross-source test pairs (genuine-in-OLX-style vs fake-in-studio-style).

---

## Documentation & References

| Resource | Description | Link |
|---|---|---|
| DINOv2 official repo | License (Apache 2.0 for standard weights), model sizes, usage | https://github.com/facebookresearch/dinov2 |
| DINOv2 linear-probe benchmark | ViT-B/14 distilled = 84.5% ImageNet top-1 frozen+linear | https://arxiv.org/html/2304.07193v2 |
| DINOv3 (Meta blog, Aug 2025) | Newer, bigger (7B), bespoke non-Apache "DINOv3 License" — not drop-in OSS | https://ai.meta.com/blog/dinov3-self-supervised-vision-model/ |
| Guo et al. 2017 | Canonical temperature-scaling paper | https://arxiv.org/pdf/1706.04599 |
| gpleiss/temperature_scaling | Reference impl — **unmaintained 8y (PyTorch 0.3 era)**, reimplement logic, don't depend on package | https://github.com/gpleiss/temperature_scaling |
| jacobgil/pytorch-grad-cam | 13k-star, active, ViT support via `blocks[-1].norm1` target layer | https://github.com/jacobgil/pytorch-grad-cam |
| GMAR paper (ViT/DINO Grad-CAM comparison) | Grad-CAM beats naive Attention Rollout on spatial precision across DINO-family models | https://arxiv.org/html/2504.19414v1 |
| PyImageSearch blur detection | Canonical Laplacian-variance blur-gate tutorial | https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ |
| BRISQUE (Mittal et al., IEEE TIP 2012) | No-reference IQA, secondary check for blur-gate's texture-confound failure mode | https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf |
| imagehash (PyPI: ImageHash) | Perceptual-hash dedup library (aHash/pHash/dHash/wHash) | https://pypi.org/project/ImageHash/ |
| hiQ Labs v. LinkedIn (9th Cir. 2022) | Scraping public data ≠ CFAA violation (doesn't clear copyright/ToS contract claims) | EFF case page / Wikipedia |
| LAION-5B precedent | Ships URLs only, not image bytes — court-tested "weights/index open, raw images not redistributed" pattern | Wikipedia / deeplearning.ai "The Batch" |
| Reddit Data API Terms (rev. 2026-07-20) | **Explicitly bans using User Content to train ML models without written permission** — Reddit has sued Anthropic and Perplexity over this | https://support.reddithelp.com (Data API Terms / Responsible Builder Policy) |
| Mercado Livre Developer ToS | Explicitly prohibits scraping/robots outside official API | https://developers.mercadolivre.com.br/pt_br/termos-e-condicoes |
| AWS EC2 G5 instance page | g5.xlarge official specs (1x A10G 24GB, 4 vCPU, 16GiB) | https://aws.amazon.com/ec2/instance-types/g5/ |
| Hoken Tech / TrustWatch | Precedent: single dial-photo authentication, 8 brands — 96.2% accuracy is 3rd-party-sourced, NOT shown on vendor's own page | https://hokentech.tech |

---

## Recommended Libraries & Tools

| Name | Purpose | Maturity | Notes |
|---|---|---|---|
| `facebook/dinov2` (HF/torch.hub) | Frozen ViT-B/14 backbone | Stable, active (Meta) | Apache 2.0 for standard checkpoints; avoid the XRay-DINO/Cell-DINO specialty checkpoints (non-commercial licenses) |
| `jacobgil/pytorch-grad-cam` | Explainability heatmaps | Stable, active, 13k stars | Not DINOv2-specifically vendor-tested — smoke-test on your actual checkpoint before trusting output |
| Guo et al. temperature-scaling logic | Calibration | Reimplement standalone | Don't `pip install` gpleiss' package as-is — it's unmaintained; the algorithm is ~15 lines (fit 1 scalar via NLL on held-out logits) |
| `ImageHash` (pip) | Perceptual-hash dedup | Stable | Use pHash variant (resize/recompress/crop-tolerant); Hamming threshold has no literature constant — tune empirically on this dataset |
| OpenCV `cv2.Laplacian` | Blur/sharpness gate | Stdlib-adjacent, stable | Threshold not universal (practitioner range 100-1000) — false-positives on flat/polished metal (watch casebacks); pair with BRISQUE if false-reject rate is high |

### Recommended Stack
DINOv2 ViT-B/14 (frozen, Apache 2.0) → per-angle linear-probe heads → rule-based angle aggregation (task already specifies, no second model) → temperature-scaling calibration (reimplemented, not the abandoned package) → jacobgil Grad-CAM for heatmaps → AWS g6.xlarge (L4, ~20% cheaper than spec'd g5.xlarge, same 24GB VRAM, irrelevant bandwidth deficit for a frozen-backbone forward-pass-dominated workload).

---

## Patterns & Best Practices

### Linear probe before fine-tune
**When to use**: always first — matches DINOv2's own published benchmark methodology (84.5% ImageNet top-1 frozen+linear for ViT-B/14).
**Trade-offs**: faster, less overfitting risk on a small (~500/class/angle) dataset; ceiling may be lower than fine-tuning. Unfreeze last blocks only if validation accuracy plateaus.

### Cross-factory / cross-source holdout, not just train/val split
**When to use**: mandatory before any weight release (task's own item 8/13 requirement, confirmed necessary by calibration literature — temperature scaling does not fix miscalibration under covariate shift).
**Trade-offs**: costs more labeled holdout data, but a model that isn't tested this way will silently fail on the next month's new superclone factory.

### ViT Grad-CAM, not naive Attention Rollout
**When to use**: every inference, for both user-facing confidence and source-bias auditing (background lighting up in the heatmap = flagrant evidence of the source-style confound).
**Trade-offs**: Grad-CAM-family gives more spatially precise, class-discriminative maps than rollout per cross-model comparison (arXiv 2504.19414); needs target-layer wiring (`blocks[-1].norm1`), not zero-config.
```python
from pytorch_grad_cam import GradCAM
target_layers = [model.blocks[-1].norm1]
cam = GradCAM(model=model, target_layers=target_layers)
```

### Blur/sharpness gate with a tunable, not hardcoded, threshold
**When to use**: before inference and before adding a scraped photo to the dataset.
**Trade-offs**: `cv2.Laplacian(img, cv2.CV_64F).var()` is a one-liner but conflates "blurry" with "low-texture" — a sharp photo of a flat polished caseback can score as low-variance as a genuinely blurry shot. Tune per angle; add BRISQUE as a secondary no-reference check if false-reject rate matters.

---

## Common Pitfalls & Solutions

| Issue | Impact | Solution |
|---|---|---|
| Model learns photo-style (OLX vs. Chrono24 lighting/background) instead of the watch | High — invalidates the whole classifier, undetectable without a specific test | Cross-source test pairs (genuine-in-OLX-style vs fake-in-studio-style); tight crop; photographic-style augmentation (task's own item 13) |
| Temperature scaling treated as "fixes calibration, done" | High — score is meaningless exactly where it matters most (novel superclone) | Explicit cross-factory holdout eval; report calibration error separately for seen vs. unseen factories |
| Depending on `gpleiss/temperature_scaling` pip package as production dependency | Low-Medium — unmaintained since PyTorch 0.3 era | Reimplement the ~15-line algorithm directly in your codebase |
| Scraping r/rolex for genuine photos without Reddit's written permission | **Critical — Reddit's Data API Terms explicitly ban ML-training use of User Content without written permission from Reddit AND rightsholder; Reddit has sued Anthropic and Perplexity over this exact pattern** | Get the dono's explicit sign-off before scraping Reddit; treat as a separate, higher-risk decision from Watchuseek/forums, not bundled in |
| Hardcoding a Laplacian-variance or perceptual-hash Hamming-distance threshold as "the standard number" | Medium — no literature-backed constant exists for either | State both as empirically-tuned-per-dataset in code comments and manifest, not as fixed universal truths |
| Citing TrustWatch's "96.2% accuracy" as vendor-confirmed | Low (reputational, not technical) if ever stated publicly | The figure is 3rd-party-directory-sourced, not shown on Hoken Tech's own product page — caveat it if cited externally |
| Assuming g5.xlarge is the cheapest adequate instance | Low (cost only) | g6.xlarge (L4, 24GB) is ~20% cheaper and has identical VRAM for this frozen-backbone workload; its lower memory bandwidth doesn't matter here |

---

## Recommendations

1. **Backbone**: DINOv2 ViT-B/14, frozen, Apache 2.0 — verified license and verified strong frozen-feature transfer (84.5% linear-probe benchmark). No direct academic precedent exists for watch/counterfeit specifically — confirmed by search, matches task's own claim.
2. **Calibration**: reimplement Guo et al. 2017 temperature scaling standalone; do not pip-depend on the unmaintained `gpleiss/temperature_scaling`. Report calibration error split by seen-factory vs. unseen-factory holdout, since temperature scaling does not fix shift-induced miscalibration.
3. **Explainability**: `jacobgil/pytorch-grad-cam` targeting `model.blocks[-1].norm1`; smoke-test against the actual DINOv2 checkpoint before trusting it in production — compatibility is architecturally plausible but not vendor-confirmed for DINOv2 specifically.
4. **Infra**: AWS g6.xlarge (L4, 24GB, ~$0.805/hr) as a cheaper drop-in for the task-specified g5.xlarge (A10G, 24GB, ~$1.006/hr) — same VRAM, irrelevant bandwidth deficit for this forward-pass-dominated frozen-backbone workload. Confirm current pricing at deploy time (sourced from 3rd-party trackers, not AWS's own price-list API).
5. **Dataset legal risk — flag before scraping**: Reddit (r/rolex) requires written permission under Reddit's own Data API Terms to use content for ML training; this is categorically different risk from OLX/Mercado Livre/Chrono24/Watchuseek and needs explicit dono sign-off, not blanket treatment as "another forum." Mercado Livre's ToS also explicitly prohibits scraping outside its official API. The "publish weights, don't publish scraped images" pattern the task already plans is a real, court-tested precedent (LAION), not just a convenient excuse — but note LAION's win leaned partly on non-commercial posture, which weakens if this project's planned future monetization (API/pro version) kicks in.
6. **Dataset quality gates**: tune the Laplacian-variance blur threshold and perceptual-hash (pHash) Hamming-distance dedup threshold empirically per this dataset — neither has a literature-backed universal constant; watch casebacks' flat polished surfaces are a known false-positive trigger for blur detection specifically.

---

## Implementation Guidance

### Installation
```bash
pip install torch==2.5.1 torchvision==0.20.1  # pin to your CUDA version
pip install transformers==4.46.0  # for facebook/dinov2-base via AutoModel
pip install grad-cam==1.5.4       # jacobgil/pytorch-grad-cam
pip install ImageHash==4.3.1      # perceptual hashing
pip install opencv-python==4.10.0.84  # cv2.Laplacian blur gate
```

### Configuration
- Backbone: `facebook/dinov2-base` (ViT-B/14, 768-dim embeddings), loaded frozen (`requires_grad_(False)` on all backbone params).
- Per-angle linear head: `nn.Linear(768, num_classes)` where `num_classes` = genuine + grade1-4 + rejection.
- Calibration: fit temperature scalar on a held-out validation split AFTER training converges, never on train data.

### Integration Points
- Angle aggregation is rule-based (max/vote across per-angle heads), not a second model — matches task's architecture decision.
- Grad-CAM output should be surfaced alongside every inference score (dual role: user trust + source-bias audit).
- Dataset manifest (task item 17) should record, per photo: source URL, scrape date, label, grade, perceptual hash, blur-gate score — needed for both reproducibility and legal audit trail.

---

## Code Examples

### Loading frozen DINOv2 ViT-B/14 + linear probe head
```python
from transformers import AutoModel
import torch.nn as nn

backbone = AutoModel.from_pretrained("facebook/dinov2-base")
for p in backbone.parameters():
    p.requires_grad = False

head = nn.Linear(768, NUM_CLASSES)  # genuine + grade1-4 + rejection
```

### Temperature scaling (reimplemented, not the abandoned package)
```python
import torch

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return temperature.item()
```

### Blur gate (tune threshold per dataset, not hardcoded as universal)
```python
import cv2

def sharpness_score(gray_image) -> float:
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# BLUR_THRESHOLD: tune empirically per angle; practitioner range is 100-1000,
# NOT a universal constant. Flat/polished casebacks score low even in focus.
```

---

## Sources & Verification

| Source | Type | Last Verified |
|---|---|---|
| github.com/facebookresearch/dinov2 | Official repo | 2026-11-03 |
| ai.meta.com/blog/dinov3-self-supervised-vision-model | Official blog | 2026-11-03 (post dated Aug 2025) |
| arxiv.org/pdf/1706.04599 (Guo et al.) | Academic paper | 2026-11-03 |
| github.com/gpleiss/temperature_scaling | GitHub repo (unmaintained) | 2026-11-03 |
| github.com/jacobgil/pytorch-grad-cam | GitHub repo | 2026-11-03 |
| arxiv.org/html/2504.19414v1 (GMAR) | Academic paper | 2026-11-03 |
| pyimagesearch.com/2015/09/07/blur-detection-with-opencv | Tutorial, cross-verified | 2026-11-03 |
| support.reddithelp.com Data API Terms | Official ToS (rev. 2026-07-20) | 2026-11-03 |
| developers.mercadolivre.com.br/pt_br/termos-e-condicoes | Official ToS | 2026-11-03 |
| aws.amazon.com/ec2/instance-types/g5 | Official AWS page | 2026-11-03 |
| hokentech.tech | Vendor site (primary, but 96.2% stat is 3rd-party not vendor-shown) | 2026-11-03 |
| devzero.io / ec2calc.cloud / economize.cloud (g5/g6 pricing) | 3rd-party trackers, cross-verified | 2026-11-03 (data labeled Oct 2026) |

---

## Changelog

| Date | Changes |
|---|---|
| 2026-11-03 | Initial creation for task: Train watch authenticity model (Rolex Submariner MVP) |
