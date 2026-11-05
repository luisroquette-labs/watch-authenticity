---
title: Treinar modelo open source de autenticação de relógios (MVP Rolex Submariner)
---

## Initial User Prompt

Treinar um modelo de visão de IA para identificar com precisão cópias e falsificações de relógios famosos. O usuário sobe fotos de ângulos específicos, pré-determinadas e guiadas, das partes de cada modelo mais propensas a entregar falsificações, e o modelo estima na hora uma chance entre 0,1% e 99,9% — nunca afirmação absoluta — de confiança para falso ou real. Antes de criar, investigar precedentes na internet ("nada se cria, tudo se copia").

### Requirements

Decisões tomadas no brainstorm (03/11/2026):

1. **Escopo MVP**: 1 modelo icônico — Rolex Submariner.
2. **Produto**: captura guiada por ângulos fixos (4-6 fotos: ciclope/lupa da data, rehaut, coroa, fecho, dial, caseback), score de confiança calibrado em camadas, nunca veredito absoluto.
3. **Graus de fake (taxonomia de mercado, vira classe do modelo)**:
   - Grau 1 — camelô/street fake (quartz, lupa sem aumento, peso errado)
   - Grau 2 — réplica comum (automático genérico, erros visíveis em foto)
   - Grau 3 — high-grade rep (factories conhecidas, erro só em detalhe fino)
   - Grau 4 — superclone (Clean, VSF — clone até do movimento)
   - Classes de treino: genuíno + graus 1-4.
4. **Score condicionado ao grau**: saída tipo "fake provável (94%), consistente com grau 2" ou "nenhum sinal detectável por foto; se fake, seria grau 3+ — envie fotos macro". Teto fixo de confiança para grau 4 por foto de celular (nunca acima de ~60% 'genuíno') — superclone não é detectável sem macro/microscopia.
5. **Dataset — raspagem rotulada pela fonte** (label vem da fonte, não de julgamento):
   - OLX/Mercado Livre anúncios declarados "réplica/inspirado" = FAKE, grau 1-2 por faixa de preço (foto de usuário real — melhor proxy do input do produto)
   - Sites chineses de réplica = FAKE (filtrar foto de estúdio/roubada de genuíno)
   - Fóruns de réplica (RepTime etc.) com factory nomeada = FAKE grau 3-4 (mapa factory → grau)
   - Chrono24 dealers verificados = GENUÍNO
   - Fóruns de donos (Watchuseek, r/rolex) = GENUÍNO (foto de pulso, usuário real)
   - OLX/ML anunciando "original" = NUNCA entra (vendedor mente, envenena o treino)
   - Filtros automáticos: deduplicação (perceptual hash), descarte de foto de vitrine (fundo branco perfeito, imagem repetida entre anúncios)
   - Alvo mínimo: ~500 fotos por classe por ângulo; recorte por parte do relógio (detector simples ou crop manual nas primeiras centenas)
6. **Arquitetura**: backbone DINOv2 ViT-B/14 congelado (pré-treinado da Meta) + cabeça de classificação por ângulo (linear probe; destravar últimas camadas só se saturar). Augmentação pesada de luz/blur/compressão JPEG. Agregação dos ângulos por regra simples (não segundo modelo).
7. **Calibração**: temperature scaling em validação separada — sem isso o score 0,1-99,9% não significa nada.
8. **Avaliação**: conjunto de teste com factories que o treino nunca viu (superclone novo chega todo mês).
9. **Infra de treino**: AWS g5.xlarge (A10G, ~US$1/h, treino 5-20h = US$5-20/rodada), créditos de bônus do dono. Fotos em S3. GitHub credits não servem (sem GPU de treino).
10. **Open source**: pesos publicados (~400 MB). Fotos do dataset NÃO publicadas (copyright/ToS de raspagem — área cinzenta; decisão pendente do dono sobre abrir dataset parcial). Monetização futura: API hospedada / versão pro.
11. **App** Next.js/Vercel com captura guiada fica para DEPOIS do modelo treinado (tarefa separada).
12. **Precedente mapeado**: TrustWatch (Hoken Tech) faz similar, fechado, com 1 foto do dial, 96,2% alegado. Entrupy usa scanner óptico proprietário. Nenhum open source, nenhum dataset público de fake vs. real. Diferencial nosso: open source + multi-ângulo guiado + graus de fake + score honesto em camadas.

Adições de arquitetura (03/11/2026):

13. **Defesa contra vício de fonte (risco nº 1)**: fakes vêm majoritariamente de fórum/OLX e genuínos de Chrono24 — o modelo pode aprender o estilo fotográfico da fonte em vez do relógio. Mitigações obrigatórias: recorte apertado (fundo fora), augmentação de estilo fotográfico, e teste cruzado de fontes (genuíno em foto "estilo OLX" vs. fake em foto de estúdio). Acurácia que despenca no teste cruzado = modelo decorou a fonte; bloqueia release.
14. **Classe de rejeição**: saída adicional "não-é-Submariner / não avaliável" para foto de outro relógio, tremida ou sem relógio. Nunca chutar score em input fora da distribuição.
15. **Porteiro de qualidade de foto**: checagem de nitidez/luz/enquadramento na captura (ex. variância do Laplaciano), antes da inferência — foto ruim é recusada com instrução de recaptura, não pontuada.
16. **Explicabilidade**: mapa de calor (Grad-CAM) junto de todo score, mostrando a região que pesou. Dupla função: confiança do usuário + auditoria de vício de fonte (mapa destacando fundo = flagrante do item 13).
17. **Manifesto do dataset + conjunto-ouro**: cada foto com fonte, URL, label, grau e hash num manifesto versionado (raspagem reproduzível/auditável). Conjunto-ouro de ~200 fotos verificadas à mão, congelado — gate de regressão obrigatório antes de publicar qualquer versão nova dos pesos.

Papel do dono: pagar GPU/storage (avisar antes de gastar), decidir escopo/publicação, validar labels por amostragem (20-30 fotos por rodada). Todo o código é do agente.

## Description

> **Required Skill**: You MUST use and analyse `watch-authenticity-cv-training` skill before doing any modification to task file or starting implementation of it!
>
> Skill location: `.claude/skills/watch-authenticity-cv-training/SKILL.md`

This task builds the training pipeline and the trained artifact for a calibrated, multi-angle, degree-aware counterfeit-detection model for the Rolex Submariner — not the end-user app. Closed competitors (TrustWatch/Hoken Tech, Entrupy) exist but publish no methodology, no dataset, and collapse their output to a single confident number; none frame their own uncertainty honestly, and none survive a change of photographic source. This task's deliverable differentiates on exactly those three axes: open source, source-trust-based labeling with an explicit grade taxonomy (genuine + 4 fake grades), and a score that is structurally forbidden from being an absolute verdict.

The people who benefit are: the future end user of the (separately-built) guided-capture app, who gets an honest, explainable signal instead of a confident guess; and the owner, who gets an open-source, differentiated asset whose credibility rests on the pipeline actually resisting its single biggest named risk — learning "which website this photo came from" instead of "is this watch fake," because fakes are scraped overwhelmingly from forum/OLX-style photos and genuines overwhelmingly from Chrono24-style photos. That risk, and the owner's own mitigation plan for it (tight crops, style augmentation, cross-source testing, a release-blocking gate), is treated here as a first-class, independently-verified requirement, not an incidental nice-to-have. A second, newly-discovered risk — Reddit's Data API Terms explicitly prohibit using r/rolex content to train ML models without Reddit's written permission, and Reddit has litigated this exact pattern against other AI companies — modifies the user's own source list: r/rolex cannot be a blanket "genuine" source the way Watchuseek and Chrono24 are; it requires a documented permission gate.

**Scope**:
- Included: source-trust-based dataset scraping/labeling pipeline with a versioned manifest and a frozen ~200-photo gold set; perceptual-hash dedup and studio-photo filtering; a frozen DINOv2 ViT-B/14 backbone with per-angle linear-probe heads and a simple (non-learned) angle-aggregation rule; heavy lighting/blur/JPEG augmentation; AWS-GPU training with an explicit cost-confirmation step; temperature-scaling calibration on held-out validation; a fixed confidence ceiling for grade-4-consistent output; confidence clipping to (0.1%, 99.9%); an out-of-distribution/low-quality rejection path with a pre-inference quality gate; Grad-CAM explainability on every prediction; three-way evaluation (standard, cross-source, held-out-factory) with a source-bias release gate; a gold-set regression gate before any weight publish; and preparation of open weights (~400MB) + model card for publication, with raw dataset photos excluded from that publication and with Reddit-sourced content excluded from training unless a documented permission record exists.
- Excluded: the Next.js/Vercel guided-capture app and its UI (a separate, later task); any watch model other than the Rolex Submariner; publishing the raw dataset photos themselves; hosted API / monetization; the owner's own actions (approving GPU/storage spend, deciding final scope/publication, manually sampling 20-30 labels per round for QA).

**User Scenarios**:
1. **Primary Flow**: the pipeline scrapes and source-labels photos, dedups/filters them, builds the versioned manifest, trains and calibrates the model, runs all three evaluation splits plus the gold-set gate, and a well-lit, correctly-framed genuine photo scores with an in-range calibrated confidence and a heatmap.
2. **Alternative Flow**: a well-lit grade-4 superclone photo returns a genuine-confidence capped at the fixed ceiling even if the model's raw internal score would otherwise be higher.
3. **Error Handling**: a blurry/badly-framed photo is rejected before inference with a recapture instruction; a photo of a different watch (or no watch) returns the rejection class instead of a fabricated score; a candidate weight version that shows source-bias (cross-source accuracy drop) or regresses on the gold set is blocked from publishing.

---

## Acceptance Criteria

**Checklist:**

| ID | Question | Category | Importance |
|----|----------|----------|------------|
| CK-1 | Does every reported authenticity confidence value fall strictly within the open interval (0.1%, 99.9%), never exactly 0% or 100%? | hard_rule | essential |
| CK-2 | Does every fake-classified prediction report an associated grade (1, 2, 3, or 4) alongside its confidence? | hard_rule | essential |
| CK-3 | Is reported genuine-confidence capped at the fixed ceiling (~60%) whenever the output is grade-4 (superclone) consistent, regardless of the model's raw score? | hard_rule | essential |
| CK-4 | Does input that is out-of-distribution (wrong watch model, no watch, unusable) return a dedicated rejection output instead of a genuine/fake score? | hard_rule | essential |
| CK-5 | Does a pre-inference photo-quality gate (sharpness/exposure/framing) reject low-quality photos with a recapture instruction before the authenticity model ever runs on them? | hard_rule | essential |
| CK-6 | Does every authenticity prediction return an accompanying Grad-CAM heatmap identifying the influential image region? | hard_rule | essential |
| CK-7 | Does the evaluation pipeline compute accuracy on a cross-source test split (genuine photographed in replica-forum style vs. fake photographed in studio style) separately from the standard same-distribution test split? | hard_rule | essential |
| CK-8 | Is there an enforced rule that blocks publishing model weights when cross-source test accuracy degrades significantly relative to standard test accuracy? | hard_rule | essential |
| CK-9 | Does evaluation include a held-out set of factories/sources never seen during training, with per-grade accuracy reported? | hard_rule | essential |
| CK-10 | Does every training photo have a corresponding versioned manifest row recording source, URL, label, grade, and perceptual hash? | hard_rule | essential |
| CK-11 | Are perceptual-hash near-duplicate photos and studio/catalog-style photos (near-perfect white background, repeated across listings) automatically excluded from the training set? | hard_rule | important |
| CK-12 | Does a frozen gold set of hand-verified photos act as an enforced regression gate that blocks publishing any new weight version that regresses against the previous published baseline? | hard_rule | essential |
| CK-13 | Are OLX/Mercado Livre listings that claim "original/genuine" excluded from the dataset entirely, never used as a genuine-label source? | hard_rule | essential |
| CK-14 | Is the model backbone a frozen pretrained DINOv2 ViT-B/14 with a separate classification head per guided angle, combined via a simple (non-learned) aggregation rule? | hard_rule | important |
| CK-15 | Is a temperature-scaling calibration step fit on a held-out validation split that is distinct from the test split, before any confidence score is reported? | hard_rule | important |
| CK-16 | Does the training pipeline apply lighting/blur/JPEG-compression augmentation to the training images? | hard_rule | important |
| CK-17 | Are open model weights (~400MB) and a model card prepared for publication while the raw dataset photos are excluded from what gets published? | principle | important |
| CK-18 | Does any part of the pipeline treat an "original"-labeled OLX/ML ad as a genuine training example? | principle | pitfall |
| CK-19 | Does the per-angle aggregation step use a second trained/learned model instead of the required simple rule? | principle | pitfall |
| CK-20 | Does training-launch tooling require an explicit cost-confirmation step before provisioning a non-trivial AWS GPU training run? | hard_rule | important |
| CK-21 | Does a crop/region-extraction step isolate each guided angle (so classification heads see the watch part, not full-frame background)? | principle | important |
| CK-22 | Is the new code free of function/logic/concept duplication across the scrape/train/calibrate/evaluate stages (e.g., one grade-mapping implementation, one confidence-clamp implementation)? | principle | important |
| CK-23 | Does every test type selected in the Test Matrix (unit / integration / smoke / property-based) have at least one corresponding test? | hard_rule | essential |
| CK-24 | Does every Test Matrix row (main + edge + error) have a corresponding test? | hard_rule | essential |
| CK-25 | Does every testable checklist item resolve to at least one real, passing test (no orphans)? | hard_rule | essential |
| CK-26 | Does every entry in Test Cases to Cover have an implemented test? | hard_rule | essential |
| CK-27 | Does the dataset pipeline exclude r/rolex (Reddit) content from training by default, requiring a documented Reddit-permission and owner sign-off record before any Reddit-sourced photo is included? | hard_rule | essential |
| CK-28 | Does Mercado Livre sourcing route through Mercado Livre's official Developer API rather than raw scraping, per its Terms of Service? | hard_rule | important |
| CK-29 | Does any part of the pipeline include a Reddit-sourced photo in the training set without a documented permission record? | principle | pitfall |

**Regular Checks:**

- [ ] No code duplication: new code does not duplicate function/logic/concept that already exists elsewhere (e.g., grade mapping, confidence clamp, and calibration logic each implemented once, not copy-pasted across scrape/train/eval stages)
- [ ] Every test type selected in the **Test Matrix** (unit / integration / smoke / property-based) has at least one corresponding test
- [ ] Every **Test Matrix** row (main + edge + error) has a corresponding test
- [ ] Every testable checklist item resolves to at least one real, passing test — no orphans
- [ ] Every entry in the **Test Cases to Cover** list has an implemented test

**Rubric:**

| Criterion | Weight |
|-----------|--------|
| Confidence Bound & Calibration Integrity | 0.15 |
| Grade-4 Superclone Ceiling Enforcement | 0.12 |
| Out-of-Distribution / Quality Rejection Handling | 0.13 |
| Explainability (Grad-CAM) Presence & Quality | 0.10 |
| Architecture Conformance | 0.12 |
| Dataset Manifest Completeness & Auditability | 0.13 |
| Generalization & Regression Evaluation Rigor | 0.25 |

**Rubric Score Definitions:**

Scale: 1-5 integers, anchor-relative — each criterion pins `score_2`/`score_4`, and 1/3/5 are placed relative to them.

### Confidence Bound & Calibration Integrity

Does every reported confidence value stay strictly within (0.1%, 99.9%), and is it the output of an actual calibration step (temperature scaling fit on held-out validation) rather than a raw, uncalibrated softmax?

Trace a sample prediction from raw logits to final reported confidence. Check whether a clamp to (0.001, 0.999) and a temperature-scaling step (with T fit on validation, not test) are both present, then place the artifact against the anchors below.

#### Anchors

**contrast**: Clips/calibrates the raw probability before reporting it, not reports the raw softmax value unclipped and uncalibrated.

**score_2**:

```text
fake_prob = probs[1]
return f"FAKE ({fake_prob*100:.0f}%)"
```

**score_4**:

```text
calibrated = temperature_scale(grade_logits, T)
genuine_prob = clip(softmax(calibrated)[GENUINE], 0.001, 0.999)
```

### Grade-4 Superclone Ceiling Enforcement

Whenever the predicted class is grade-4 (superclone) consistent, is the reported genuine-confidence capped at the fixed ~60% ceiling regardless of the model's raw score?

Feed a synthetic case where the raw model output for a grade-4-consistent sample implies high genuine-confidence. Check whether the output is clamped to the ceiling, then place the artifact against the anchors below.

#### Anchors

**contrast**: Applies the fixed grade-4 genuine-confidence ceiling, not leaves the raw score unclamped.

**score_2**:

```text
return f"GENUINE ({(1 - fake_prob)*100:.0f}%)"   # no grade, no ceiling
```

**score_4**:

```text
if grade == GRADE_4:
    genuine_prob = min(genuine_prob, GRADE4_GENUINE_CEILING)  # ~0.60
```

### Out-of-Distribution / Quality Rejection Handling

Does the pipeline return a dedicated rejection output (not a fabricated genuine/fake score) for input that is blurry/badly framed, or not a Submariner / no watch at all — and does the quality gate run BEFORE the authenticity model, not after?

Trace the control flow for a blurry input and for a wrong-subject input. Confirm both short-circuit to a rejection output before any authenticity scoring occurs, then place the artifact against the anchors below.

#### Anchors

**contrast**: Short-circuits to a Rejection before scoring, not always forces a FAKE/GENUINE verdict regardless of input validity.

**score_2**:

```text
if fake_prob > 0.5:
    return f"FAKE ({fake_prob*100:.0f}%)"
else:
    return f"GENUINE (...)"
```

**score_4**:

```text
if quality < MIN_SHARPNESS:
    return Rejection("blurry", "recapture needed")
if not is_submariner(image_path):
    return Rejection("not-a-submariner", "unsupported model")
```

### Explainability (Grad-CAM) Presence & Quality

Does every authenticity prediction return a Grad-CAM heatmap identifying the image region that most influenced the score, and is the heatmap non-degenerate (not all-zero / not uniformly flat) on a real trained checkpoint?

Check whether the prediction return type includes a heatmap field, and inspect one real heatmap from a trained checkpoint for non-degeneracy, then place the artifact against the anchors below.

#### Anchors

**contrast**: Returns a heatmap alongside the score, not returns only the score with no explainability artifact.

**score_2**:

```text
return f"FAKE ({fake_prob*100:.0f}%)"
```

**score_4**:

```text
heatmap = grad_cam(model, image_path)
return Prediction(genuine_prob, grade, heatmap)
```

### Architecture Conformance

Is the backbone a frozen pretrained DINOv2 ViT-B/14 with a separate classification head per guided angle, aggregated by a simple, non-learned rule, and trained with lighting/blur/JPEG augmentation — as explicitly specified, rather than an unspecified or freely-substituted architecture?

Inspect the model-building code: which parameters are frozen, how per-angle heads are structured, what the aggregation function is (and whether it is itself a trained module), and whether augmentation transforms are wired into the training data pipeline. Reference pattern: `.claude/skills/watch-authenticity-cv-training/SKILL.md` ("Loading frozen DINOv2 ViT-B/14 + linear probe head", ViT-compatible Grad-CAM target `model.blocks[-1].norm1`). Place the artifact against the anchors below.

#### Anchors

**contrast**: Builds per-angle heads over a frozen backbone and aggregates by a simple rule, not has no angle structure or frozen-backbone signal at all.

**score_2**:

```text
logits = model(img)   # single undifferentiated forward pass, no angle structure, no backbone-freeze signal
```

**score_4**:

```text
grade_logits = aggregate_angles(angle_scores)  # simple rule, no learned aggregator
# backbone built elsewhere with requires_grad=False; per-angle linear heads on top
```

### Dataset Manifest Completeness & Auditability

Does every training photo's manifest row record source, URL, label, grade and perceptual hash (enabling audit/reproducibility without redistributing the image), and are hash-duplicates, studio/catalog photos, "original"-labeled OLX/ML ads, and unpermitted Reddit content excluded before training?

Inspect a sample of manifest rows for field completeness, and inspect the ingestion filters for dedup/studio-photo/"original"-ad/Reddit-permission exclusion logic, then place the artifact against the anchors below.

#### Anchors

**contrast**: Manifest row carries source/url/grade/phash for audit, not carries only a bare path+label.

**score_2**:

```text
path,label
img001.jpg,fake
```

**score_4**:

```text
{"path": "img001.jpg", "source": "reptime_forum", "url": "https://reptime.example/thread/1234",
 "label": "fake", "grade": 3, "phash": "a1b2c3d4e5f6", "scraped_at": "2026-11-03"}
```

### Generalization & Regression Evaluation Rigor

Does the evaluation report standard-test, cross-source-test, AND held-out-factory-test accuracy (not just one aggregate number), and does a gold-set regression gate plus a cross-source-degradation gate actually block publishing when either fails?

Read the evaluation report/harness output for a candidate weight version. Check whether all three accuracy breakdowns are present and whether the publish path is wired to refuse publishing on gold-set regression or cross-source degradation (not just print a warning), then place the artifact against the anchors below.

#### Anchors

**contrast**: Reports and gates on standard + cross-source + held-out-factory + gold-set results, not reports one aggregate accuracy number with no gating.

**score_2**:

```text
Test accuracy: 94.2%
```

**score_4**:

```text
Standard test accuracy: 94.2%
Cross-source test accuracy: 91.8%  (delta 2.4pp, within 10pp gate)
Held-out-factory test accuracy: grade1 96% | grade2 93% | grade3 81% | grade4 62%
Gold-set regression check vs. v1 baseline: PASS
```

**Test Strategy:**

**Criticality:** HIGH

**Test Matrix:**

| Type | Size | Framework | Dependencies | Gate |
|------|------|-----------|--------------|------|
| unit | small | pytest (or equivalent Python unit framework) | — | Gate 1 |
| integration | medium | pytest + local fixture HTML/image files + tmpdir | fixture HTML pages, tmpdir filesystem for manifest/checkpoints, small fixture image set for a 1-epoch smoke training run | Gate 2 |
| smoke | medium | pytest script loading the exported artifact | exported model weights file, one gold-set sample image | Gate 5 |
| property-based | small | hypothesis (or equivalent Python property-based framework) | — | Gate 6 |

**Test Cases to Cover**

#### CK-1: Does every reported authenticity confidence value fall strictly within (0.1%, 99.9%)?
- [unit] clip_confidence(0.0) returns 0.001 [BVA: below floor]
- [unit] clip_confidence(1.0) returns 0.999 [BVA: above ceiling]
- [unit] clip_confidence(0.5) returns 0.5 unchanged [EP: typical mid-range]
- [property-based] for any raw probability p in [0,1], clip_confidence(p) is in [0.001, 0.999]

#### CK-2: Does every fake-classified prediction report a grade (1-4)?
- [unit] classify(known_grade2_sample) returns fake=true with grade=2 [EP: typical fake]
- [unit] classify(known_genuine_sample) returns fake=false with no grade field [EP: typical genuine]

#### CK-3: Is genuine-confidence capped at ~60% for grade-4-consistent output?
- [unit] apply_grade4_ceiling(raw_genuine=0.95, grade=4) returns <= 0.60 [EP: raw above ceiling]
- [unit] apply_grade4_ceiling(raw_genuine=0.40, grade=4) returns 0.40 unchanged [EP: raw below ceiling]
- [property-based] for any raw_genuine in [0,1], apply_grade4_ceiling(raw_genuine, grade=4) <= 0.60

#### CK-4: Does OOD/wrong-subject input return the rejection class, not a score?
- [unit] classify(photo_of_different_watch_model) returns Rejection("not-a-submariner")
- [unit] classify(no_watch_in_frame) returns Rejection("not-evaluable")

#### CK-5: Does a pre-inference quality gate reject bad photos before the authenticity model runs?
- [unit] quality_gate(sharp_well_lit_fixture_image) passes
- [unit] quality_gate(blurry_fixture_image_below_laplacian_threshold) rejects with recapture instruction [BVA around threshold: thresh-1 / thresh / thresh+1]
- [integration] submitting a blurry fixture photo through the full pipeline results in zero calls to the authenticity model

#### CK-6: Does every prediction carry a Grad-CAM heatmap?
- [unit] predict(valid_fixture_image) response includes a heatmap array matching image dimensions
- [integration] a small-scale trained checkpoint produces a non-degenerate (not all-zero) Grad-CAM heatmap for a real fixture sample image

#### CK-7: Does evaluation compute a cross-source test split separately from the standard test split?
- [integration] evaluation harness computes both standard-test and cross-source-test accuracy from the manifest-labeled fixture set

#### CK-8: Does an enforced rule block publish on cross-source degradation?
- [unit] publish_gate(standard_acc=0.94, cross_source_acc=0.60, max_delta=0.10) blocks publish [EP: large degradation]
- [unit] publish_gate(standard_acc=0.94, cross_source_acc=0.91, max_delta=0.10) allows publish [BVA: delta at/just under threshold]

#### CK-9: Does evaluation include a factory-disjoint held-out set with per-grade accuracy?
- [integration] evaluation harness reports per-grade accuracy on a held-out split whose factories/sources are absent from the training manifest

#### CK-10: Does every training photo have a manifest row with source, url, label, grade, hash?
- [unit] manifest_row(fixture_photo) includes non-empty source, url, label, grade, phash fields [EP: well-formed]

#### CK-11: Are hash-duplicate and studio/catalog photos excluded automatically?
- [unit] dedup_filter([imgA, imgA_near_duplicate]) drops the duplicate by perceptual hash distance below threshold
- [unit] studio_photo_filter(white_background_catalog_fixture) flags the image for exclusion
- [property-based] hash_distance(a, b) == hash_distance(b, a) for any two fixture images [symmetry invariant]

#### CK-12: Does the gold-set regression gate block publish on regression?
- [integration] publish script refuses to publish when gold-set accuracy on the fixture gold set regresses vs. a frozen baseline recorded for the previous version

#### CK-13: Are "original"-labeled OLX/ML ads excluded from the dataset entirely?
- [unit] source_label_rule(listing_text="original", marketplace="olx") returns EXCLUDED, never GENUINE

#### CK-14: Is the backbone frozen DINOv2 with per-angle heads and simple aggregation?
- [unit] build_model() backbone parameters have requires_grad=False; only per-angle head parameters are trainable
- [unit] aggregate_angles(scores_per_angle) uses the documented simple rule and contains no trainable parameters

#### CK-15: Is temperature scaling fit on a held-out validation split distinct from test?
- [unit] temperature_scale(logits, T=2.0) yields a flatter (lower max) probability than T=1.0 for the same logits [EP: typical]
- [property-based] for any logit vector and T>1, max(softmax(logits/T)) <= max(softmax(logits)) [monotonicity invariant]
- [integration] calibration-fit step reads only the validation split's manifest rows, never the test split's

#### CK-16: Does training apply lighting/blur/JPEG augmentation?
- [unit] augment_pipeline(fixture_image) applies at least one of {blur, jpeg-compression, lighting} and returns a tensor different from the unaugmented input

#### CK-20: Does training-launch tooling require explicit cost confirmation?
- [unit] launch_training(instance_type="g5.xlarge", confirm=False) raises/refuses to provision
- [unit] launch_training(instance_type="g5.xlarge", confirm=True) proceeds to provisioning

#### CK-21: Does crop/region-extraction isolate each guided angle?
- [unit] crop_angle(fixture_full_frame_image, angle="rehaut") returns a region smaller than the full frame centered on the expected sub-region

#### CK-23/CK-24/CK-25/CK-26: Strategy realization and coverage
- [smoke] exported weights artifact loads and returns a schema-valid prediction (confidence in range, grade/rejection field present, heatmap present) for one fixture gold-set image

#### CK-27: Does the pipeline exclude Reddit (r/rolex) content without documented permission?
- [unit] source_label_rule(source="reddit", subreddit="r/rolex", permission_on_file=False) returns EXCLUDED
- [unit] source_label_rule(source="reddit", subreddit="r/rolex", permission_on_file=True) returns GENUINE [EP: documented-permission path]

#### CK-28: Does Mercado Livre sourcing use the official API, not raw scraping?
- [unit] mercadolivre_ingest(method="scrape") raises/refuses to ingest
- [unit] mercadolivre_ingest(method="official_api") proceeds to ingest [EP: compliant path]

**Definition of Done:**

- [ ] Every `essential` checklist item answers YES
- [ ] All Regular Checks pass
- [ ] Every test case in **Test Cases to Cover** is implemented and passing
- [ ] Dataset manifest exists, is versioned, and every training photo has a complete row (source, url, label, grade, phash)
- [ ] The ~200-photo gold set exists, is frozen, and the regression-gate check passes for the version being delivered
- [ ] Standard-test, cross-source-test, and held-out-factory-test evaluation reports are produced, and the cross-source degradation gate passes (or publish is explicitly blocked with the result reported)
- [ ] Model weights (~400MB) and a model card are prepared for open-source publication; raw dataset photos are excluded from that publication
- [ ] No Reddit-sourced photo is present in the training set without a documented permission record; Mercado Livre sourcing uses the official API
- [ ] Any non-trivial AWS GPU training run was preceded by an explicit cost confirmation

---

## Architecture

> **References**: Skill `.claude/skills/watch-authenticity-cv-training/SKILL.md` · Analysis `.specs/analysis/train-watch-authenticity-model-impact.md` · Scratchpad `.specs/scratchpad/9d22fd32.md`

### Solution Strategy

**Approach**: Greenfield Python pipeline built as a **Clean Architecture core wrapped in a Pipeline/Batch flow with Ports & Adapters** for external systems. Every safety-critical, rubric-scrutinized rule — confidence clamp, grade-4 ceiling, grade mapping, source-label rule, non-learned angle aggregation, publish/regression gate predicates, manifest record schema — lives **once** in a framework-free `core/` domain package (no `torch`/`opencv`/`boto3` imports). The ordered stages (scrape → build dataset → train → calibrate → evaluate → publish) are use-cases that orchestrate adapters implementing the SKILL.md "Recommended Stack" verbatim. This structure is the single highest-leverage decision for this task: it makes the hard-rule invariants impossible to duplicate (CK-22) or bypass, and trivially unit/property-testable with no fixtures.

**Key Decisions**:
1. **Pure domain core (`core/`)**: all safety logic isolated, framework-free — because the task's credibility rests on a few invariants the rubric tests directly, and because CK-22 forbids duplicating them across stages.
2. **Adapter layer = SKILL.md Recommended Stack unchanged**: frozen `facebook/dinov2-base` → per-angle `nn.Linear(768, NUM_CLASSES)` → rule-based aggregation → reimplemented Guo et al. temperature scaling → `jacobgil/pytorch-grad-cam` targeting `model.blocks[-1].norm1`.
3. **Legal/cost gates at the adapter boundary**: Reddit ingestion defaults `EXCLUDED` unless a documented permission record exists (CK-27/29); Mercado Livre exposes only an official-API ingest path (CK-28); AWS provisioning requires `confirm=True` (CK-20).
4. **Three-way evaluation with publish-blocking gates**: standard + cross-source + held-out-factory accuracy, plus gold-set regression gate — all wired to *refuse* publish, not warn (CK-7/8/9/12).

**Trade-offs Accepted**:
- One extra layer (`core/`) over a flat script — accepted to buy single-source-of-truth invariants and zero-fixture testability.
- Default to **g6.xlarge** (L4, ~$0.805/h, 24GB) over the spec's g5.xlarge — ~20% cheaper, identical VRAM for this frozen-backbone workload; the divergence is surfaced to the owner at the cost-confirmation step rather than resolved silently.

### Architecture Decisions

#### AD-1: Clean Architecture core + Pipeline + Ports & Adapters

**Status**: Accepted

**Context**: Greenfield repo; task credibility depends on a small set of safety invariants the rubric tests directly, and CK-22 forbids duplicating them.

**Options**: (1) Flat module-per-concern tree (analysis map); (2) Single monolithic script; (3) Clean-core + concern-organized adapter/use-case stages (hybrid).

**Decision**: Option 3 — pure framework-free `core/` holding every safety rule once, with concern directories as the adapter/use-case layer.

**Consequences**:
- Invariants (clamp, ceiling, grade-map, aggregation, gates) implemented once; CK-22 structurally satisfied.
- `core/` is `torch`-free → unit + property tests need no GPU and no fixtures.
- Slightly more structure than a script; every stage must import from `core/` rather than re-implementing.

#### AD-2: g6.xlarge as default training instance

**Status**: Accepted (pending owner confirm at spend time)

**Context**: Spec item 9 says g5.xlarge (~$1/h); SKILL.md recommends g6.xlarge (~$0.805/h), same 24GB VRAM.

**Options**: (1) g5.xlarge as specified; (2) g6.xlarge cheaper counter-pick.

**Decision**: Default g6.xlarge; surface the g5-vs-g6 choice to the owner at the mandatory cost-confirmation step (CK-20), never silently.

**Consequences**: ~20% cheaper per round; divergence from spec is explicit and owner-gated; confirm live pricing at deploy time (SKILL.md sources it from 3rd-party trackers).

#### AD-3: Reddit permission gate separated from other "genuine" sources

**Status**: Accepted

**Context**: Reddit Data API Terms explicitly ban ML-training use of User Content without written permission; Reddit has litigated this pattern. Spec item 5 listed r/rolex alongside Watchuseek as a routine genuine source.

**Options**: (1) Treat r/rolex like any forum; (2) Default-exclude Reddit, require documented permission + owner sign-off.

**Decision**: Option 2 — `source_label_rule` returns `EXCLUDED` for Reddit unless `permission_on_file=True`.

**Consequences**: Legal exposure removed by default; genuine photos rely on Watchuseek + Chrono24 absent permission; a documented-permission path exists but is off by default (CK-27/29).

### Architecture Decomposition

**Components**:

| Component | File Path | Responsibility | Dependencies | Reuses From |
|---|---|---|---|---|
| Domain core | `core/types.py`, `core/rules.py` | Grade/Label enums, `ManifestRecord`, `Prediction`/`Rejection` VOs, pure rules: `clip_confidence`, `apply_grade4_ceiling`, `grade_map`, `source_label_rule`, `aggregate_angles`, `publish_gate`, `goldset_regression_gate` | stdlib only | New — task-specific business rules (none in SKILL.md) |
| Scrapers | `scrapers/<source>.py` (olx, mercadolivre, chrono24, reptime, watchuseek, reddit, chinese_replica) | Fetch photos per source; derive label via `core.source_label_rule` | `core`, `requests`/official SDKs | New; ML uses official API only; Reddit default-excluded |
| Manifest/labeling | `labeling/manifest.py` | Versioned manifest I/O; one grade-mapping call site into `core` | `core` | New |
| Dataset ops | `dataset/dedup.py`, `dataset/crop.py`, `dataset/augment.py`, `dataset/splits.py` | pHash dedup + studio filter; per-angle crop; lighting/blur/JPEG aug; train/val/test/cross-source/factory-holdout/gold splits | `ImageHash`, `opencv`, `core` | New |
| Backbone adapter | `adapters/backbone.py` | Load frozen `facebook/dinov2-base` (`requires_grad=False`) | `transformers` | **SKILL.md:146-150** (as-is) |
| Per-angle heads | `model/heads.py` | `nn.Linear(768, NUM_CLASSES)` per guided angle | `torch`, backbone | **SKILL.md:150** |
| Rejection/OOD | `model/rejection.py` | `classify(photo)` → `Rejection("not-a-submariner"/"not-evaluable")` for OOD | `core`, heads | New |
| Calibration | `calibration/temperature_scaling.py` | `fit_temperature` on held-out val; wrap with `core.apply_grade4_ceiling` + `core.clip_confidence` | `torch`, `core` | **SKILL.md:157-168** (as-is), extended with ceiling |
| Quality gate | `quality_gate/blur_light_check.py` | `quality_gate(image)` pre-inference sharpness/exposure/framing; `sharpness_score` | `opencv` | **SKILL.md:174-175** (as-is), per-angle threshold |
| Explainability | `explainability/gradcam.py` | Grad-CAM heatmap targeting `model.blocks[-1].norm1` on every prediction | `grad-cam`, `torch` | **SKILL.md:81-82** (as-is), smoke-test on real ckpt |
| Inference | `inference/predict.py` | Orchestrate quality→rejection→backbone→heads→aggregate→calibrate→clamp→ceiling→gradcam → `Prediction` | all above + `core` | New (orchestration) |
| Evaluation | `evaluation/release_gate.py`, `evaluation/evaluate.py` | 3-split accuracy + per-grade; call `core.publish_gate` + `core.goldset_regression_gate` | `core`, inference | New |
| Infra | `infra/aws_train.(sh|py)`, `infra/s3_sync.py` | GPU provisioning behind `confirm=True`; S3 photo sync | `boto3` | New; g6.xlarge default (AD-2) |
| Release | `release/publish_weights.py`, model card | Publish weights (~400MB) + card only if gates PASS; exclude raw photos | `huggingface_hub`, `core` | New |
| Project meta | `requirements.txt`, `pyproject.toml`/preflight (ruff+mypy+pytest) | Pinned deps + CI preflight (none exists yet) | — | **SKILL.md:120-124** pins |

**Interactions**:

```
scrapers ──► labeling/manifest ──► dataset(dedup/crop/augment/splits) ──► train ──► calibration
    │              │                        │                                           │
    └──► core.source_label_rule            core                                        core
                                                                                         │
 inference/predict ──► quality_gate ─(reject)─► Rejection                                │
        │                                                                                │
        ├─► model.rejection ─(OOD)─► Rejection                                           │
        └─► backbone+heads ─► core.aggregate_angles ─► calibration ─► core.clip_confidence
                                                              └─► core.apply_grade4_ceiling ─► explainability.gradcam ─► Prediction
 evaluation ──► core.publish_gate + core.goldset_regression_gate ──► release.publish_weights
```

### Building Block View

```
┌──────────────────────────────────────────────────────────────┐
│                     watch-authenticity                         │
│  L4 Frameworks: torch · transformers · opencv · ImageHash ·    │
│                 grad-cam · boto3 · huggingface_hub             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ L3 Adapters: scrapers/ · adapters/backbone · model/*    │  │
│  │              calibration · quality_gate · explainability │  │
│  │              infra/* · release/*                          │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ L2 Use-Cases (stages): scrape · build_dataset ·   │   │  │
│  │  │     train · calibrate · evaluate · publish        │   │  │
│  │  │   ┌──────────────────────────────────────────┐   │   │  │
│  │  │   │ L1 core/ (pure, framework-free)          │   │   │  │
│  │  │   │  types: Grade · Label · ManifestRecord · │   │   │  │
│  │  │   │         Prediction · Rejection           │   │   │  │
│  │  │   │  rules: clip_confidence · apply_grade4_  │   │   │  │
│  │  │   │   ceiling · grade_map · source_label_    │   │   │  │
│  │  │   │   rule · aggregate_angles · publish_gate │   │   │  │
│  │  │   │   · goldset_regression_gate              │   │   │  │
│  │  │   └──────────────────────────────────────────┘   │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
Dependencies point inward only: core imports nothing above it.
```

### High-Level Structure

```
Feature: Rolex Submariner authenticity model (training pipeline + trained artifact)
├── Entry Points: stage CLIs (scrape, build-dataset, train, calibrate, evaluate, publish) + inference/predict
├── Core Logic: core/ pure invariants (clamp, ceiling, grade-map, aggregation, gates, schema)
├── Data Layer: versioned manifest + frozen gold set + S3 photo store + exported weights (~400MB)
└── Output: calibrated per-angle prediction {confidence∈(0.1%,99.9%), grade|rejection, ceiling_applied, heatmap} + 3-split eval report + open weights + model card
```

### Workflow Steps

```
Training pipeline:
1. scrape ──► 2. source-label ──► 3. dedup + studio-filter ──► 4. crop-per-angle
      │             │                     │                          │
      ▼             ▼                     ▼                          ▼
 raw photos   label+grade          deduped set               per-angle crops
                                                                     │
5. manifest rows ──► 6. splits (train/val/test/cross-source/factory-holdout/gold)
      │                     │
      ▼                     ▼
 versioned manifest    disjoint splits
                            │
7. augment ──► 8. train per-angle heads (frozen backbone) ──► 9. fit temperature on VAL
                                                                     │
10. evaluate 3 splits + gold ──► 11. publish_gate + goldset_regression_gate
                                        │
                              PASS ─────┴───── FAIL ──► publish blocked, result reported
                                │
                                ▼
                     12. publish weights + model card (raw photos excluded)
```

### Runtime Scenarios

**Scenario: Primary — well-lit genuine photo**
```
User photo ──► quality_gate(pass) ──► submariner_check(pass) ──► backbone+heads
                                                                      │
   Prediction{ conf∈(0.1%,99.9%), grade=none(genuine), heatmap } ◄── aggregate ─► calibrate ─► clip
```

**Scenario: Alternative — grade-4 superclone, well-lit**
```
photo ──► gates(pass) ──► model ──► aggregate ──► calibrate ──► clip
                                                        │
          genuine_prob = min(raw_genuine, 0.60 ceiling) ◄── grade==GRADE_4
                                                        │
                   Prediction{ conf≤60%, grade=4, ceiling_applied=true, heatmap }
```

**Scenario: Error handling**
```
blurry photo ──► quality_gate(fail) ──► Rejection("blurry","recapture")   [0 calls to authenticity model]
other watch ──► quality_gate(pass) ──► submariner_check(fail) ──► Rejection("not-a-submariner")
candidate weights w/ cross-source drop OR gold-set regression ──► publish_gate(fail) ──► publish BLOCKED
```

**State transitions (publish)**:
```
[candidate weights] ─ eval 3 splits ─► [evaluated] ─ gates pass ─► [publishable] ─► [published]
                                            │
                                     gate fails
                                            ▼
                                   [blocked, reported]
```

### Contracts

**Manifest record** (one row per training photo; CK-10):
```
ManifestRecord
├── path: str (required)
├── source: str (required)         # e.g. reptime_forum, chrono24, mercadolivre_api
├── url: str (required)
├── label: Label (required)        # genuine | fake | excluded
├── grade: int|null                # 1-4 when fake, null when genuine
├── phash: str (required)          # perceptual hash
├── scraped_at: date (required)
└── blur_score: float (optional)
```

**Prediction / Rejection output** (CK-1..6):
```
Prediction
├── confidence: float   # strictly in (0.001, 0.999)
├── genuine: bool
├── grade: int|null     # 1-4 present when fake (CK-2)
├── ceiling_applied: bool
└── heatmap: array      # Grad-CAM, image-dimensioned, non-degenerate (CK-6)

Rejection
├── reason: str         # "blurry" | "not-a-submariner" | "not-evaluable"
└── instruction: str    # e.g. "recapture needed"
```

**Core rule signatures** (framework-free; each implemented once — CK-22):
```python
clip_confidence(p: float) -> float                  # → [0.001, 0.999]
apply_grade4_ceiling(raw_genuine: float, grade: int) -> float   # grade4 → ≤0.60
grade_map(factory_or_price: ...) -> int             # single grade-mapping impl
source_label_rule(source, *, subreddit=None, marketplace=None,
                  listing_text=None, permission_on_file=False) -> Label
aggregate_angles(scores_per_angle) -> dict          # simple rule, 0 trainable params (CK-19)
publish_gate(standard_acc, cross_source_acc, max_delta) -> bool
goldset_regression_gate(current_acc, baseline_acc) -> bool
```

**Adapter signatures** (external-system gates):
```python
fit_temperature(logits, labels) -> float            # SKILL.md:157-168, val split only
quality_gate(image) -> Pass | Rejection             # pre-inference, SKILL.md:174-175
mercadolivre_ingest(method: Literal["official_api"]) -> ...   # "scrape" raises (CK-28)
launch_training(instance_type, *, confirm: bool)    # confirm=False refuses (CK-20)
```

### Expected Changes

```
watch-authenticity/
├── core/
│   ├── types.py                     # NEW: Grade/Label enums, ManifestRecord, Prediction, Rejection
│   └── rules.py                     # NEW: clip_confidence, apply_grade4_ceiling, grade_map,
│                                     #      source_label_rule, aggregate_angles, publish_gate,
│                                     #      goldset_regression_gate (all pure, torch-free)
├── scrapers/<source>.py             # NEW: per-source ingest; mercadolivre=official API only; reddit default-excluded
├── labeling/manifest.py             # NEW: versioned manifest I/O
├── dataset/{dedup,crop,augment,splits}.py   # NEW
├── adapters/backbone.py             # NEW: frozen DINOv2 loader
├── model/{heads,rejection}.py       # NEW
├── calibration/temperature_scaling.py       # NEW: fit_temperature + ceiling wrap
├── quality_gate/blur_light_check.py # NEW
├── explainability/gradcam.py        # NEW
├── inference/predict.py             # NEW: inference orchestration
├── evaluation/{evaluate,release_gate}.py     # NEW
├── infra/{aws_train,s3_sync}.*      # NEW: g6.xlarge default, confirm-gated
├── release/publish_weights.py       # NEW: weights+card, raw photos excluded
├── tests/                           # NEW: unit/integration/smoke/property suite
├── requirements.txt                 # NEW: pinned deps (SKILL.md:120-124)
└── pyproject.toml / preflight       # NEW: ruff + mypy + pytest (no CI exists yet)
```

### Build Sequence

```
Phase 1: Foundation (core + project meta) — everything imports core
- [ ] requirements.txt with pinned deps
  - Reuse: SKILL.md:120-124 version pins
- [ ] core/types.py (enums, ManifestRecord, Prediction, Rejection)
  - Reuse: None — task-specific value objects
- [ ] core/rules.py: clip_confidence, apply_grade4_ceiling, grade_map, source_label_rule,
      aggregate_angles, publish_gate, goldset_regression_gate (pure, torch-free)
  - Reuse: None — task-specific business rules; implemented ONCE here (CK-22)
- [ ] pyproject/preflight (ruff + mypy + pytest)
  - Reuse: None — no CI convention exists in repo

Phase 2: Dataset ingestion (depends on Phase 1 core)
- [ ] scrapers/* with label derivation via core.source_label_rule; mercadolivre official-API-only; reddit default-excluded
  - Reuse: core.source_label_rule
- [ ] labeling/manifest.py
  - Reuse: core.ManifestRecord, core.grade_map
- [ ] dataset/dedup.py (pHash + studio filter), dataset/crop.py (per-angle)
  - Reuse: ImageHash, opencv
- [ ] dataset/augment.py (lighting/blur/JPEG), dataset/splits.py (6 splits incl. gold)
  - Reuse: None — new; enforce no leak across splits

Phase 3: Model + calibration (depends on Phase 1-2)
- [ ] adapters/backbone.py frozen DINOv2
  - Reuse: SKILL.md:146-150 (as-is)
- [ ] model/heads.py per-angle nn.Linear(768,N); model/rejection.py OOD
  - Reuse: SKILL.md:150
- [ ] calibration/temperature_scaling.py (fit on val) + wrap core.apply_grade4_ceiling + core.clip_confidence
  - Reuse: SKILL.md:157-168 fit_temperature; core ceiling+clamp

Phase 4: Inference path (depends on Phase 3)
- [ ] quality_gate/blur_light_check.py
  - Reuse: SKILL.md:174-175 sharpness_score; threshold tuned per angle
- [ ] explainability/gradcam.py targeting blocks[-1].norm1
  - Reuse: SKILL.md:81-82; smoke-test on real checkpoint
- [ ] inference/predict.py orchestration (quality→rejection→model→aggregate→calibrate→clamp→ceiling→gradcam)
  - Reuse: core.aggregate_angles, core.clip_confidence, core.apply_grade4_ceiling

Phase 5: Evaluation + release (depends on Phase 4)
- [ ] evaluation/evaluate.py (standard + cross-source + held-out-factory, per-grade)
  - Reuse: inference/predict
- [ ] evaluation/release_gate.py calling core.publish_gate + core.goldset_regression_gate
  - Reuse: core gate predicates
- [ ] release/publish_weights.py (weights+card, raw photos excluded; gated on PASS)
  - Reuse: core gates

Phase 6: Infra (independent of 3-5; needed before real training run)
- [ ] infra/aws_train.* behind confirm=True (g6.xlarge default, surface g5-vs-g6 to owner)
  - Reuse: None — new; AD-2
- [ ] infra/s3_sync.py
  - Reuse: boto3
```

---

## Implementation Process

You MUST launch for each step a separate agent, instead of performing all steps yourself. And for each step marked as parallel, you MUST launch separate agents in parallel.

**CRITICAL:** For each agent you MUST:
1. Use the **Model** and **Agent** type specified in the step's sub-task file (e.g., `haiku`, `sonnet`, `sdd:developer`)
2. Provide the path to THIS task file AND the path to that step's sub-task file
3. Require agent to implement exactly that step, not more, not less, not other steps

**CRITICAL:** Verification is done at PHASE level, not per step. When every step of a phase is complete, you MUST launch the code reviewer ONCE for that phase, at the **Reviewer model** named for that phase in the Phase Overview.

### Parallelization Overview

```
Phase 1
  Step 01-project-scaffolding [haiku] ───┐ (parallel, width 2)
  Step 02-core-types [sonnet] ───────────┘
                    │ (02)
                    ▼
  Step 03-core-rules [sonnet]  (Needs 02)
         ═══ end of Phase 1 (review: opus) ═══
                    │ (core ready)
      ┌─────────────┴─────────────┐
      ▼                           ▼
Phase 2                        (infra lane)
  Step 04a-manifest [sonnet]   Step 04b-infra [sonnet]   (parallel, width 2; need core)
      │                           │
      ├───────────┬───────────┐   │
      ▼           ▼           ▼   │
  05a-scrapers 05b-dedup-   05c-augment-   (parallel, width 3; need 04a-manifest)
  [sonnet]     crop[sonnet]  splits[sonnet]
      └───────────┴───────────┴───┘
         ═══ end of Phase 2 (review: opus) ═══
                    │ (05c dataset splits)
                    ▼
Phase 3
  Step 06-backbone-heads [sonnet]  (Needs 05c + core)
                    │
          ┌─────────┴─────────┐  (parallel, width 2)
          ▼                   ▼
  07a-rejection[sonnet]  07b-calibration[sonnet]
          └─────────┬─────────┘
         ═══ end of Phase 3 (review: opus) ═══
                    │
Phase 4
      ┌─────────────┴─────────────┐  (parallel, width 2)
      ▼                           ▼
  08a-quality-gate[sonnet]   08b-gradcam[sonnet]
  (needs core)               (needs 06)
      └─────────────┬─────────────┘
                    ▼
  Step 09-inference-predict [sonnet]  (Needs 06, 07a, 07b, 08a, 08b)
         ═══ end of Phase 4 (review: opus) ═══
                    │
Phase 5
  Step 10-evaluate [sonnet]        (Needs 09)
                    │
  Step 11-release-gate [sonnet]    (Needs 10)
                    │
  Step 12-publish-weights [sonnet] (Needs 11)
         ═══ end of Phase 5 (review: opus) ═══
```

| Step | Phase | Model | Agent | Depends on | Parallel with | Sub-Task File |
|------|-------|-------|-------|------------|---------------|---------------|
| `01-project-scaffolding` | Phase 1 | haiku | haiku | None | `02-core-types` | `.specs/sub-tasks/train-watch-authenticity-model.feature/01-project-scaffolding.md` |
| `02-core-types` | Phase 1 | sonnet | sdd:developer | None | `01-project-scaffolding` | `.specs/sub-tasks/train-watch-authenticity-model.feature/02-core-types.md` |
| `03-core-rules` | Phase 1 | sonnet | sdd:developer | `02-core-types` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/03-core-rules.md` |
| `04a-manifest` | Phase 2 | sonnet | sdd:developer | `03-core-rules` | `04b-infra` | `.specs/sub-tasks/train-watch-authenticity-model.feature/04a-manifest.md` |
| `04b-infra` | Phase 2 | sonnet | sdd:developer | `03-core-rules` | `04a-manifest` | `.specs/sub-tasks/train-watch-authenticity-model.feature/04b-infra.md` |
| `05a-scrapers` | Phase 2 | sonnet | sdd:developer | `04a-manifest` | `05b-dedup-crop`, `05c-augment-splits` | `.specs/sub-tasks/train-watch-authenticity-model.feature/05a-scrapers.md` |
| `05b-dedup-crop` | Phase 2 | sonnet | sdd:developer | `04a-manifest` | `05a-scrapers`, `05c-augment-splits` | `.specs/sub-tasks/train-watch-authenticity-model.feature/05b-dedup-crop.md` |
| `05c-augment-splits` | Phase 2 | sonnet | sdd:developer | `04a-manifest` | `05a-scrapers`, `05b-dedup-crop` | `.specs/sub-tasks/train-watch-authenticity-model.feature/05c-augment-splits.md` |
| `06-backbone-heads` | Phase 3 | sonnet | sdd:developer | `05c-augment-splits` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/06-backbone-heads.md` |
| `07a-rejection` | Phase 3 | sonnet | sdd:developer | `06-backbone-heads` | `07b-calibration` | `.specs/sub-tasks/train-watch-authenticity-model.feature/07a-rejection.md` |
| `07b-calibration` | Phase 3 | sonnet | sdd:developer | `06-backbone-heads` | `07a-rejection` | `.specs/sub-tasks/train-watch-authenticity-model.feature/07b-calibration.md` |
| `08a-quality-gate` | Phase 4 | sonnet | sdd:developer | `03-core-rules` | `08b-gradcam` | `.specs/sub-tasks/train-watch-authenticity-model.feature/08a-quality-gate.md` |
| `08b-gradcam` | Phase 4 | sonnet | sdd:developer | `06-backbone-heads` | `08a-quality-gate` | `.specs/sub-tasks/train-watch-authenticity-model.feature/08b-gradcam.md` |
| `09-inference-predict` | Phase 4 | sonnet | sdd:developer | `06-backbone-heads`, `07a-rejection`, `07b-calibration`, `08a-quality-gate`, `08b-gradcam` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/09-inference-predict.md` |
| `10-evaluate` | Phase 5 | sonnet | sdd:developer | `09-inference-predict` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/10-evaluate.md` |
| `11-release-gate` | Phase 5 | sonnet | sdd:developer | `10-evaluate` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/11-release-gate.md` |
| `12-publish-weights` | Phase 5 | sonnet | sdd:developer | `11-release-gate` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/12-publish-weights.md` |

### Phase Overview

#### Phase 1: Foundation (pure core + preflight)

Steps: `01-project-scaffolding`, `02-core-types`, `03-core-rules`
Reviewer model: `opus`
Acceptance Criteria that should be fulfiled:
Checklist items:
- `CK-1` — Does every reported confidence fall strictly within (0.1%, 99.9%)? (`clip_confidence` defined + tested)
- `CK-3` — Is genuine-confidence capped at ~60% for grade-4-consistent output? (`apply_grade4_ceiling` defined + tested)
- `CK-8` — Is there a rule that blocks publish on cross-source degradation? (`publish_gate` predicate)
- `CK-12` — Does the gold-set regression gate block publishing a regressed version? (`goldset_regression_gate` predicate)
- `CK-13` — Are OLX/ML "original" listings excluded, never a genuine source? (`source_label_rule`)
- `CK-18` — Does any part treat an "original"-labeled OLX/ML ad as genuine? (pitfall — must be NO)
- `CK-19` — Does aggregation use a learned model instead of the simple rule? (pitfall — `aggregate_angles` has 0 trainable params)
- `CK-22` — Is the code free of duplicated grade-map/clamp/gate logic? (single-source `core/rules.py`)
- `CK-27` — Does the pipeline exclude r/rolex without a documented permission record? (`source_label_rule`)
- `CK-29` — Does any part include Reddit content without a permission record? (pitfall — must be NO)

Rubrics:
- `Confidence Bound & Calibration Integrity` (clamp portion)
- `Grade-4 Superclone Ceiling Enforcement` (ceiling function)

#### Phase 2: Dataset ingestion (+ training infra)

Steps: `04a-manifest`, `04b-infra`, `05a-scrapers`, `05b-dedup-crop`, `05c-augment-splits`
Reviewer model: `opus`
Acceptance Criteria that should be fulfiled:
Checklist items:
- `CK-10` — Does every training photo have a manifest row with source, URL, label, grade, phash?
- `CK-11` — Are perceptual-hash near-duplicates and studio/catalog photos auto-excluded?
- `CK-13` — Are OLX/ML "original" listings excluded at ingestion?
- `CK-16` — Does training apply lighting/blur/JPEG augmentation?
- `CK-18` — Does ingestion treat an "original" OLX/ML ad as genuine? (pitfall — must be NO)
- `CK-20` — Does training-launch tooling require an explicit cost-confirmation step?
- `CK-21` — Does a crop/region-extraction step isolate each guided angle?
- `CK-27` — Does the scraper exclude r/rolex without a documented permission record?
- `CK-28` — Does Mercado Livre sourcing route through the official Developer API, not raw scraping?
- `CK-29` — Does any scraper include Reddit content without a permission record? (pitfall — must be NO)

Rubrics:
- `Dataset Manifest Completeness & Auditability`

#### Phase 3: Model + calibration

Steps: `06-backbone-heads`, `07a-rejection`, `07b-calibration`
Reviewer model: `opus`
Acceptance Criteria that should be fulfiled:
Checklist items:
- `CK-14` — Is the backbone a frozen DINOv2 ViT-B/14 with a per-angle head and simple non-learned aggregation?
- `CK-15` — Is temperature scaling fit on a held-out validation split distinct from test?
- `CK-3` — Is the grade-4 ceiling applied inside the calibration output path?

Rubrics:
- `Architecture Conformance`
- `Confidence Bound & Calibration Integrity` (temperature-scaling portion)

#### Phase 4: Inference path

Steps: `08a-quality-gate`, `08b-gradcam`, `09-inference-predict`
Reviewer model: `opus`
Acceptance Criteria that should be fulfiled:
Checklist items:
- `CK-1` — Does every reported confidence fall strictly within (0.1%, 99.9%) at runtime?
- `CK-2` — Does every fake-classified prediction report an associated grade (1-4)?
- `CK-3` — Is genuine-confidence capped at the ceiling for grade-4-consistent output?
- `CK-4` — Does OOD input (wrong watch / no watch) return a rejection, not a score?
- `CK-5` — Does a pre-inference quality gate reject bad photos before the authenticity model runs?
- `CK-6` — Does every prediction return an accompanying Grad-CAM heatmap?

Rubrics:
- `Out-of-Distribution / Quality Rejection Handling`
- `Explainability (Grad-CAM) Presence & Quality`
- `Grade-4 Superclone Ceiling Enforcement` (applied in output)

#### Phase 5: Evaluation + release

Steps: `10-evaluate`, `11-release-gate`, `12-publish-weights`
Reviewer model: `opus`
Acceptance Criteria that should be fulfiled:
Checklist items:
- `CK-7` — Does evaluation compute a cross-source test split separately from the standard split?
- `CK-8` — Is publish blocked when cross-source accuracy degrades significantly?
- `CK-9` — Does evaluation include a held-out factory set with per-grade accuracy?
- `CK-12` — Does the frozen gold set act as an enforced regression gate that blocks publish?
- `CK-17` — Are open weights (~400MB) + model card prepared while raw dataset photos are excluded?
- `CK-23` — Does every Test Matrix type (unit/integration/smoke/property-based) have a test?
- `CK-24` — Does every Test Matrix row (main + edge + error) have a test?
- `CK-25` — Does every testable checklist item resolve to at least one passing test (no orphans)?
- `CK-26` — Does every entry in Test Cases to Cover have an implemented test?

Rubrics:
- `Generalization & Regression Evaluation Rigor`
- `Dataset Manifest Completeness & Auditability` (raw photos excluded from publish)
