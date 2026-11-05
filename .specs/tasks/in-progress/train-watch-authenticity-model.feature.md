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
5. **Dataset — aquisição e rótulo separados** (decisão refinada em 05/11/2026):
   - Fonte, texto do vendedor, preço e factory declarada geram apenas candidato/rótulo proposto; nunca ground truth por si sós.
   - Nível A: exame físico, autenticação/serviço ou evidência controlada; Nível B: transação/factory explícita ou declaração autenticada de parceiro ligada ao relógio; ambos podem treinar.
   - Nível C: reputação da fonte, anúncio, preço ou relato não corroborado; fica em descoberta/revisão e não treina. Conflito ou identidade insuficiente = inconclusivo.
   - OLX/ML anunciando "original" nunca prova genuinidade; Reddit continua excluído sem permissões; Mercado Livre usa somente API oficial.
   - Direitos válidos são exigidos antes do download e antes do congelamento; raw/evidências permanecem privados no COS.
   - Alvo mínimo por classe/ângulo: 500 imagens elegíveis, 100 relógios físicos distintos, diversidade de fontes/vendedores e divisão por grupos sem vazamento.
6. **Arquitetura**: backbone DINOv2 ViT-B/14 congelado (pré-treinado da Meta) + cabeça de classificação por ângulo (linear probe; destravar últimas camadas só se saturar). Augmentação pesada de luz/blur/compressão JPEG. Agregação dos ângulos por regra simples (não segundo modelo).
7. **Calibração**: temperature scaling em validação separada — sem isso o score 0,1-99,9% não significa nada.
8. **Avaliação**: conjunto de teste com factories que o treino nunca viu (superclone novo chega todo mês).
9. **Infra de treino (decisão substituída em 05/11/2026)**: IBM Cloud VPC no perfil `gx3-16x80x1l4` (1x NVIDIA L4 24 GB, 16 vCPU, 80 GiB RAM), usando o crédito promocional de US$ 18.000 da CF GAUSS SERVICOS LTDA. Fotos, manifestos, checkpoints e relatórios ficam no IBM Cloud Object Storage; a imagem reprodutível de treino fica no IBM Cloud Container Registry. AWS/S3 ficam removidos do caminho normal e não existe fallback pago automático.
10. **Open source**: pesos publicados (~400 MB). Fotos do dataset NÃO publicadas (copyright/ToS de raspagem — área cinzenta; decisão pendente do dono sobre abrir dataset parcial). Monetização futura: API hospedada / versão pro.
11. **App** Next.js/Vercel com captura guiada fica para DEPOIS do modelo treinado (tarefa separada).
12. **Precedente mapeado**: TrustWatch (Hoken Tech) faz similar, fechado, com 1 foto do dial, 96,2% alegado. Entrupy usa scanner óptico proprietário. Nenhum open source, nenhum dataset público de fake vs. real. Diferencial nosso: open source + multi-ângulo guiado + graus de fake + score honesto em camadas.

Adições de arquitetura (03/11/2026):

13. **Defesa contra vício de fonte (risco nº 1)**: fakes vêm majoritariamente de fórum/OLX e genuínos de Chrono24 — o modelo pode aprender o estilo fotográfico da fonte em vez do relógio. Mitigações obrigatórias: recorte apertado (fundo fora), augmentação de estilo fotográfico, e teste cruzado de fontes (genuíno em foto "estilo OLX" vs. fake em foto de estúdio). Acurácia que despenca no teste cruzado = modelo decorou a fonte; bloqueia release.
14. **Classe de rejeição**: saída adicional "não-é-Submariner / não avaliável" para foto de outro relógio, tremida ou sem relógio. Nunca chutar score em input fora da distribuição.
15. **Porteiro de qualidade de foto**: checagem de nitidez/luz/enquadramento na captura (ex. variância do Laplaciano), antes da inferência — foto ruim é recusada com instrução de recaptura, não pontuada.
16. **Explicabilidade**: mapa de calor (Grad-CAM) junto de todo score, mostrando a região que pesou. Dupla função: confiança do usuário + auditoria de vício de fonte (mapa destacando fundo = flagrante do item 13).
17. **Manifesto do dataset + conjunto-ouro**: cada foto com fonte, URL, label, grau e hash num manifesto versionado (raspagem reproduzível/auditável). Conjunto-ouro de ~200 fotos verificadas à mão, congelado — gate de regressão obrigatório antes de publicar qualquer versão nova dos pesos.
18. **IBM-first com limite de utilidade (05/11/2026)**: usar serviços IBM quando substituírem uma responsabilidade real do pipeline: VPC GPU para treino, Cloud Object Storage para dados/artefatos, Container Registry para a imagem, IAM/trusted profile para acesso sem segredo persistente, e Logs/Monitoring + billing/usage para auditoria e proteção de saldo. Não adicionar watsonx, Kubernetes, banco ou outro serviço apenas para consumir crédito; o modelo e os pesos continuam portáveis e open source.
19. **Estado do crédito**: código aplicado segundo confirmação do dono em 05/11/2026. Saldo, data terminal e elegibilidade de cada SKU permanecem `PENDENTE E2E` até leitura autoritativa em `Billing and usage > Promotions and credits`. Nenhum provisionamento pode ocorrer antes dessa prova e de uma estimativa inferior ao saldo com margem para atraso de faturamento.
20. **Aquisição operacional e direitos (05/11/2026)**: APIs oficiais, exports de parceiros e uploads autorizados geram recibos imutáveis. Fonte desconhecida, direito ausente/expirado/revogado ou endpoint ilustrativo falham fechados antes da rede. Cada ativo conserva hashes, batch, relógio físico, cluster de repost, vendedor, estilo de captura, evidência e direitos.
21. **Escala por referência (05/11/2026)**: o primeiro artefato continua restrito ao Submariner. Nova referência reutiliza o backbone, mas só recebe cabeça/calibração próprias após passar os mesmos gates de evidência, entidade, classe/ângulo, fonte, gold e holdout; volume bruto não autoriza cobertura.

Papel do dono: decidir escopo/publicação e validar labels por amostragem (20-30 fotos por rodada). O crédito IBM financia GPU/storage; qualquer risco de overage fora do saldo exige nova autorização explícita. Todo o código é do agente.

## Description

> **Required Skill**: You MUST use and analyse `watch-authenticity-cv-training` skill before doing any modification to task file or starting implementation of it!
>
> Skill location: `.claude/skills/watch-authenticity-cv-training/SKILL.md`

This task builds the training pipeline and the trained artifact for a calibrated, multi-angle, degree-aware counterfeit-detection model for the Rolex Submariner — not the end-user app. Closed competitors (TrustWatch/Hoken Tech, Entrupy) exist but publish no methodology, no dataset, and collapse their output to a single confident number; none frame their own uncertainty honestly, and none survive a change of photographic source. This task's deliverable differentiates on exactly those three axes: open source, evidence-gated labeling with an explicit grade taxonomy (genuine + 4 fake grades), and a score that is structurally forbidden from being an absolute verdict.

The people who benefit are: the future end user of the (separately-built) guided-capture app, who gets an honest, explainable signal instead of a confident guess; and the owner, who gets an open-source, differentiated asset whose credibility rests on the pipeline actually resisting its single biggest named risk — learning "which website this photo came from" instead of "is this watch fake." The mitigation is not only crop/augmentation: source reputation cannot create ground truth, every training row needs A/B evidence and rights, physical-watch/repost/seller/batch groups cannot cross splits, and cross-source/temporal/factory tests block release. Reddit remains excluded without written permissions; Watchuseek and Chrono24 may supply candidates but are not blanket genuine sources.

**Scope**:
- Included: rights-gated candidate acquisition, A/B/C/inconclusive evidence review, immutable receipts, physical-watch/repost/seller/batch grouping, a versioned manifest and a frozen ≥200-photo Tier-A gold set; perceptual-hash dedup and capture-style classification; a frozen DINOv2 ViT-B/14 backbone with per-angle linear-probe heads and a simple (non-learned) angle-aggregation rule; heavy lighting/blur/JPEG augmentation; IBM Cloud VPC L4 training with credit/balance/cost/TTL gates; IBM Cloud Object Storage, Container Registry, IAM and job observability; temperature-scaling calibration on held-out validation; a fixed confidence ceiling for grade-4-consistent output; confidence clipping to (0.1%, 99.9%); an out-of-distribution/low-quality rejection path with a pre-inference quality gate; Grad-CAM explainability on every prediction; standard, cross-source, held-out-factory, temporal and gold evaluation with release gates; and preparation of open weights (~400MB) + model card, excluding raw dataset/evidence documents.
- Excluded: the Next.js/Vercel guided-capture app and its UI (a separate, later task); any watch model other than the Rolex Submariner; publishing the raw dataset photos themselves; hosted API / monetization; the owner's own actions (approving GPU/storage spend, deciding final scope/publication, manually sampling 20-30 labels per round for QA).

**User Scenarios**:
1. **Primary Flow**: an approved API/export/upload creates immutable candidates; rights and evidence review admit only A/B records; grouping/dedup/readiness freeze leakage-safe splits; training/calibration runs all evaluation gates; and a well-lit, correctly-framed genuine photo scores with an in-range calibrated confidence and a heatmap.
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
| CK-11 | Are near-duplicates clustered, repeated/uncertain catalog photos excluded, and rights-cleared confirmed studio photos retained only in their correct train/evaluation role? | hard_rule | important |
| CK-12 | Does a frozen gold set of hand-verified photos act as an enforced regression gate that blocks publishing any new weight version that regresses against the previous published baseline? | hard_rule | essential |
| CK-13 | Are OLX/Mercado Livre "original/genuine" claims excluded from every training/evaluation/gold split, with any retained metadata remaining candidate-only? | hard_rule | essential |
| CK-14 | Is the model backbone a frozen pretrained DINOv2 ViT-B/14 with a separate classification head per guided angle, combined via a simple (non-learned) aggregation rule? | hard_rule | important |
| CK-15 | Is a temperature-scaling calibration step fit on a held-out validation split that is distinct from the test split, before any confidence score is reported? | hard_rule | important |
| CK-16 | Does the training pipeline apply lighting/blur/JPEG-compression augmentation to the training images? | hard_rule | important |
| CK-17 | Are open model weights (~400MB) and a model card prepared for publication while the raw dataset photos are excluded from what gets published? | principle | important |
| CK-18 | Does any part of the pipeline treat an "original"-labeled OLX/ML ad as a genuine training example? | principle | pitfall |
| CK-19 | Does the per-angle aggregation step use a second trained/learned model instead of the required simple rule? | principle | pitfall |
| CK-20 | Does training-launch tooling refuse IBM GPU provisioning unless promo-credit balance/expiry/SKU eligibility are freshly evidenced, the estimated run fits inside available credit with safety margin, `confirm=True`, and a mandatory TTL/teardown path are all present? | hard_rule | essential |
| CK-30 | Is the training/data path IBM-native (VPC L4 + Cloud Object Storage + Container Registry + IAM + observability) with no AWS/S3 or third-party paid fallback? | hard_rule | important |
| CK-21 | Does a crop/region-extraction step isolate each guided angle (so classification heads see the watch part, not full-frame background)? | principle | important |
| CK-22 | Is the new code free of function/logic/concept duplication across the scrape/train/calibrate/evaluate stages (e.g., one grade-mapping implementation, one confidence-clamp implementation)? | principle | important |
| CK-23 | Does every test type selected in the Test Matrix (unit / integration / smoke / property-based) have at least one corresponding test? | hard_rule | essential |
| CK-24 | Does every Test Matrix row (main + edge + error) have a corresponding test? | hard_rule | essential |
| CK-25 | Does every testable checklist item resolve to at least one real, passing test (no orphans)? | hard_rule | essential |
| CK-26 | Does every entry in Test Cases to Cover have an implemented test? | hard_rule | essential |
| CK-27 | Does the pipeline refuse Reddit bytes before download unless written Reddit and rightsholder permissions plus owner authorization are recorded? | hard_rule | essential |
| CK-28 | Does Mercado Livre sourcing route through Mercado Livre's official Developer API rather than raw scraping, per its Terms of Service? | hard_rule | important |
| CK-29 | Does any part download or train on Reddit content without all required permission records? | principle | pitfall |
| CK-31 | Does every training asset have a valid rights record plus Tier A/B evidence tied to that specific candidate? | hard_rule | essential |
| CK-32 | Can source reputation, seller text or price alone ever create an accepted genuine/fake/grade label? | principle | pitfall |
| CK-33 | Are physical-watch, repost, seller and acquisition-batch groups kept within one split family? | hard_rule | essential |
| CK-34 | Does dataset freeze fail when class/angle/entity/source/gold readiness thresholds are unmet? | hard_rule | essential |
| CK-35 | Does rights expiry/revocation block reuse, produce impacted-artifact and disposition reports, and require observed quarantine/deletion without mutating prior freezes? | hard_rule | essential |
| CK-36 | Does a new watch reference require its own passing readiness/gold/holdout admission before receiving a head or coverage claim? | hard_rule | important |

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

Does every training photo record source, candidate/asset hashes, accepted label/grade, evidence tier, rights, physical-watch/repost/seller/batch groups and capture style, while excluding ineligible duplicates/catalog images, original-claim marketplace ads and unpermitted Reddit content?

Inspect candidate, rights, evidence and training records plus the frozen split assignments. Verify that a source or seller claim alone cannot create an accepted label and that all leakage groups remain within one split family.

#### Anchors

**contrast**: Accepted training rows carry rights/evidence/entity lineage, not only a source-derived path and label.

**score_2**:

```text
path,label
img001.jpg,fake
```

**score_4**:

```text
{"asset_id": "sha256:...", "candidate_id": "cand-1", "source": "partner-export",
 "label": "fake", "grade": 3, "evidence_tier": "b", "rights_record_id": "rights-1",
 "watch_entity_id": "watch-7", "content_cluster_id": "cluster-2", "capture_style": "studio"}
```

### Generalization & Regression Evaluation Rigor

Does evaluation report standard, cross-source, held-out-factory, temporal and gold results by class/angle/source, and do source-bias, evidence and regression gates block publishing when any required slice fails?

Read the evaluation report for the exact frozen dataset hash. Check every required slice plus the learning-curve and source-shortcut diagnostics; warnings without publish refusal do not pass.

#### Anchors

**contrast**: Reports and gates on standard + cross-source + held-out-factory + temporal + gold results, not one aggregate accuracy number.

**score_2**:

```text
Test accuracy: 94.2%
```

**score_4**:

```text
Standard test accuracy: 94.2%
Cross-source test accuracy: 91.8%  (delta 2.4pp, within 10pp gate)
Held-out-factory test accuracy: grade1 96% | grade2 93% | grade3 81% | grade4 62%
Temporal holdout accuracy: 88.7%
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

#### CK-10: Does every training photo have complete acquisition/evidence/rights/entity lineage?
- [unit] an accepted `TrainingRecord` includes source, candidate/asset hashes, label/grade, evidence tier, rights and all four leakage-group IDs

#### CK-11: Are duplicates/catalog photos and confirmed studio photos handled without leakage or data loss?
- [unit] dedup_filter([imgA, imgA_near_duplicate]) drops the duplicate by perceptual hash distance below threshold
- [unit] repeated catalog image with uncertain provenance is excluded; rights-cleared confirmed studio replica is retained and tagged `capture_style="studio"`
- [property-based] hash_distance(a, b) == hash_distance(b, a) for any two fixture images [symmetry invariant]

#### CK-12: Does the gold-set regression gate block publish on regression?
- [integration] publish script refuses to publish when gold-set accuracy on the fixture gold set regresses vs. a frozen baseline recorded for the previous version

#### CK-13: Are "original"-labeled OLX/ML ads excluded from the dataset entirely?
- [unit] propose_source_label(listing_text="original", marketplace="olx") returns None and cannot produce `TrainingRecord`

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
- [unit] launch_training(profile="gx3-16x80x1l4", confirm=False) raises before any IBM API call
- [unit] missing/stale credit evidence, ineligible SKU, estimate above usable balance, or missing TTL each blocks provisioning even when `confirm=True`
- [integration] a fully mocked eligible run provisions once, uploads terminal artifacts to COS, and deletes the VSI in `finally`

#### CK-30: Is the operational path IBM-native without paid fallback?
- [unit] COS upload/download uses the IBM COS client and the training image references IBM Cloud Container Registry
- [static] no runtime import/call to AWS `boto3`, EC2 or S3 remains after migration
- [integration] IAM/trusted-profile credentials are resolved without embedding an API key in source or artifacts

#### CK-21: Does crop/region-extraction isolate each guided angle?
- [unit] crop_angle(fixture_full_frame_image, angle="rehaut") returns a region smaller than the full frame centered on the expected sub-region

#### CK-23/CK-24/CK-25/CK-26: Strategy realization and coverage
- [smoke] exported weights artifact loads and returns a schema-valid prediction (confidence in range, grade/rejection field present, heatmap present) for one fixture gold-set image

#### CK-27: Does the pipeline exclude Reddit (r/rolex) content without documented permission?
- [unit] Reddit without both required permissions is refused before download
- [unit] permitted Reddit content creates a candidate only; evidence review still decides training eligibility

#### CK-28: Does Mercado Livre sourcing use the official API, not raw scraping?
- [unit] Mercado Livre raw scrape is refused before network access
- [unit] official API may create candidates but cannot write accepted labels directly

#### CK-31/CK-32: Are rights and evidence required instead of source-as-truth?
- [unit] Tier C source reputation, seller text and price cannot build a training record
- [unit] valid Tier A/B evidence plus current rights can build the matching training record

#### CK-33/CK-34: Are grouped splits and quantitative readiness enforced?
- [property-based] no physical-watch, repost, seller or batch group ID appears in two splits
- [boundary] 499 images or 99 physical watches in a required class/angle cell blocks freeze; 500/100 passes that cell

#### CK-35/CK-36: Are revocation and new-reference admission fail-closed?
- [integration] revoked rights exclude affected assets, enumerate impacted checkpoints, deny normal reads and leave deletion pending until COS absence is observed, without mutating the old freeze
- [unit] a photo-rich but source-dominated new reference cannot receive a model head or coverage claim

**Definition of Done:**

- [ ] Every `essential` checklist item answers YES
- [ ] All Regular Checks pass
- [ ] Every test case in **Test Cases to Cover** is implemented and passing
- [ ] Candidate, rights, evidence and training ledgers exist; every training asset is Tier A/B, currently authorized and group-linked
- [ ] The ≥200-photo Tier-A gold set exists, spans every output class, is frozen, and passes the regression gate
- [ ] Standard, cross-source, held-out-factory, temporal and gold reports are bound to one freeze hash; every required release gate passes or publishing is blocked
- [ ] Model weights (~400MB) and a model card are prepared for open-source publication; raw dataset photos are excluded from that publication
- [ ] No Reddit byte is downloaded without required permissions; Mercado Livre uses only its official API; neither source bypasses evidence review
- [ ] Every IBM GPU run has persisted preflight evidence (credit balance/expiry/SKU eligibility/estimate/TTL), terminal artifact upload, and VSI deletion; no overage or non-IBM paid fallback occurred

---

## Architecture

> **References**: Skill `.claude/skills/watch-authenticity-cv-training/SKILL.md` · Analysis `.specs/analysis/train-watch-authenticity-model-impact.md` · Scratchpad `.specs/scratchpad/9d22fd32.md`

### Solution Strategy

**Approach**: Greenfield Python pipeline built as a **Clean Architecture core wrapped in a Pipeline/Batch flow with Ports & Adapters** for external systems. Safety rules live once in framework-free `core/`; acquisition separates source proposals from accepted evidence labels. The ordered stages are approve source/rights → collect immutable candidates → review evidence → group/dedup → freeze ready dataset → train/calibrate/evaluate/publish. IBM ownership remains isolated so neither acquisition nor training can bypass COS integrity or credit/TTL/teardown gates.

**Key Decisions**:
1. **Pure domain core (`core/`)**: all safety logic isolated, framework-free — because the task's credibility rests on a few invariants the rubric tests directly, and because CK-22 forbids duplicating them across stages.
2. **Adapter layer = SKILL.md Recommended Stack unchanged**: frozen `facebook/dinov2-base` → per-angle `nn.Linear(768, NUM_CLASSES)` → rule-based aggregation → reimplemented Guo et al. temperature scaling → `jacobgil/pytorch-grad-cam` targeting `model.blocks[-1].norm1`.
3. **Evidence/legal/cost gates at boundaries**: source policy and rights fail closed before download; A/B evidence is required before training; IBM provisioning independently requires fresh credit evidence, eligible SKU, conservative estimate, `confirm=True`, TTL and teardown.
4. **Five-view evaluation with publish-blocking gates**: standard + cross-source + held-out-factory + temporal + gold, with per-class/angle/source reporting and source-shortcut diagnostics.

**Trade-offs Accepted**:
- One extra layer (`core/`) over a flat script — accepted to buy single-source-of-truth invariants and zero-fixture testability.
- Default to IBM profile **`gx3-16x80x1l4`** (L4, 24 GB). It is larger on CPU/RAM than the former AWS g6.xlarge, but spends the existing promotional balance and removes new cash cost while preserving the required GPU memory.

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

#### AD-2: IBM Cloud as the exclusive funded training plane

**Status**: Accepted; supersedes the AWS decision from 03/11/2026

**Context**: The owner has an applied US$ 18,000 IBM Cloud promotional credit and wants this project to consume IBM services without new cash spend. IBM documents `gx3-16x80x1l4` with one NVIDIA L4 24 GB GPU, matching the VRAM class selected for the frozen DINOv2 workload.

**Options**: (1) keep AWS g6; (2) IBM VPC L4; (3) force managed watsonx training.

**Decision**: Option 2. Use IBM VPC L4 for custom PyTorch, COS for data/artifacts, Container Registry for the immutable training image, IAM/trusted profiles for access, and IBM Logs/Monitoring plus billing/usage evidence for operations. Do not force watsonx when it duplicates the portable PyTorch pipeline.

**Consequences**: AWS code and tests must be replaced and Phase 2 re-reviewed. Promotional credit is not a hard spending cap: missing/stale balance or SKU eligibility fails closed, every VSI has TTL plus `finally` teardown, and a safety reserve protects against delayed usage reporting. No paid fallback.

#### AD-3: Source acquisition separated from accepted ground truth

**Status**: Accepted

**Context**: The prototype mapped source names directly to labels, making dealer/forum reputation look like proof and allowing the model to learn source style. Reddit adds a separate permission prohibition.

**Options**: (1) Keep source-as-label; (2) Treat sources as candidate producers and require rights plus A/B evidence for accepted labels.

**Decision**: Option 2. `propose_source_label` can only propose; evidence adjudication owns final labels. Reddit is refused before download without both required permissions.

**Consequences**: Watchuseek, Chrono24, seller text and price cannot independently train the model. Candidate volume may be lower, but label integrity, rights and leakage groups become auditable.

### Architecture Decomposition

**Components**:

| Component | File Path | Responsibility | Dependencies | Reuses From |
|---|---|---|---|---|
| Domain core | `core/types.py`, `core/rules.py`, `acquisition/types.py` | Model vocabulary plus immutable candidate/rights/evidence/training/freeze contracts and pure proposals/gates | stdlib only | Extended by `04c-data-acquisition.md` |
| Source governance/acquisition | `acquisition/source_policy.py`, `acquisition/collect.py`, `acquisition/partner_import.py`, `scrapers/<source>.py` | Default-deny source/rights decisions; immutable collection receipts; adapters parse approved API/feed/export data but never accept labels | `requests`, `infra.ibm_cos` | Reworks prototype scrapers |
| Evidence/manifest | `labeling/review.py`, `labeling/adjudicate.py`, `labeling/manifest.py` | A/B/C/inconclusive review, append-only adjudication and accepted training rows | acquisition/core contracts | Reworks flat source-labeled manifest |
| Dataset ops | `dataset/entities.py`, `dataset/dedup.py`, `dataset/crop.py`, `dataset/augment.py`, `dataset/readiness.py`, `dataset/splits.py`, `dataset/freeze.py` | repost/watch/seller/batch grouping, derivatives, augmentation, quantitative readiness and grouped standard/cross-source/factory/temporal/gold splits | ImageHash, Pillow/opencv, core | Reworks row-level split/studio deletion |
| Backbone adapter | `adapters/backbone.py` | Load frozen `facebook/dinov2-base` (`requires_grad=False`) | `transformers` | **SKILL.md:146-150** (as-is) |
| Per-angle heads | `model/heads.py` | `nn.Linear(768, NUM_CLASSES)` per guided angle | `torch`, backbone | **SKILL.md:150** |
| Rejection/OOD | `model/rejection.py` | `classify(photo)` → `Rejection("not-a-submariner"/"not-evaluable")` for OOD | `core`, heads | New |
| Calibration | `calibration/temperature_scaling.py` | `fit_temperature` on held-out val; wrap with `core.apply_grade4_ceiling` + `core.clip_confidence` | `torch`, `core` | **SKILL.md:157-168** (as-is), extended with ceiling |
| Quality gate | `quality_gate/blur_light_check.py` | `quality_gate(image)` pre-inference sharpness/exposure/framing; `sharpness_score` | `opencv` | **SKILL.md:174-175** (as-is), per-angle threshold |
| Explainability | `explainability/gradcam.py` | Grad-CAM heatmap targeting `model.blocks[-1].norm1` on every prediction | `grad-cam`, `torch` | **SKILL.md:81-82** (as-is), smoke-test on real ckpt |
| Inference | `inference/predict.py` | Orchestrate quality→rejection→backbone→heads→aggregate→calibrate→clamp→ceiling→gradcam → `Prediction` | all above + `core` | New (orchestration) |
| Evaluation | `evaluation/release_gate.py`, `evaluation/evaluate.py` | five-view slice metrics, learning curves and source-shortcut diagnostics; call publish/regression gates | `core`, inference, frozen dataset | Extended by `04c` Task 7 |
| IBM infra | `infra/ibm_types.py`, `infra/ibm_evidence.py`, `infra/ibm_cost_guard.py`, `infra/ibm_cos.py`, `infra/ibm_vpc.py`, `infra/ibm_train.py`, `infra/ibm_watchdog.py` | Authoritative promo/price capture plus usage normalization, typed spend gate, VPC L4 lifecycle, append-only COS events, trusted-profile auth, independent TTL teardown and observability | IBM VPC/COS/platform SDKs | Replaces `aws_train.py`/`s3_sync.py`; AD-2 |
| Training producer | `training/data.py`, `training/run.py` | Validate immutable COS dataset/splits, map the six classes and produce checkpoint/metrics/class map/run result | `core`, dataset, heads, torch | Extends the existing fixture-only smoke trainer |
| Release | `release/publish_weights.py`, model card | Publish weights (~400MB) + card only if gates PASS; exclude raw photos | `huggingface_hub`, `core` | New |
| Project meta | `requirements.txt`, `pyproject.toml`/preflight (ruff+mypy+pytest) | Pinned deps + CI preflight (none exists yet) | — | **SKILL.md:120-124** pins |

**Interactions**:

```
approved API/export/upload ─► rights gate ─► immutable candidates ─► evidence review
                                                                  │
                                                                  ▼
COS raw/derivatives ◄── receipts/hashes ── entity/repost grouping ─► readiness/freeze ─► train ─► calibration
                                                                                              │
                                                                                             core
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
│         grad-cam · IBM VPC/COS SDKs · huggingface_hub          │
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
├── Entry Points: approve/collect/review/freeze/train/calibrate/evaluate/publish CLIs + inference/predict
├── Core Logic: pure model plus acquisition/evidence/rights/readiness invariants
├── Data Layer: candidate/evidence/rights ledgers + frozen grouped splits + private IBM COS + exported weights
└── Output: calibrated prediction + five-view slice report + open weights/model card (no raw photos/evidence)
```

### Workflow Steps

```
Training pipeline:
1. approve source/rights ─► 2. collect immutable candidate ─► 3. review evidence
                                                                  │
4. group repost/watch/seller/batch ◄───────────────────────────────┘
      │
5. sanitize + crop/style/dedup ─► 6. readiness gate + grouped freeze
                                           │
7. augment ─► 8. train per-angle heads ─► 9. fit temperature on VAL
                                           │
10. evaluate standard/cross-source/factory/temporal/gold ─► 11. release gates
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

**Accepted training record** (candidate, rights and evidence ledgers are separate and immutable; CK-10/31/33):
```
TrainingRecord
├── asset_id/candidate_id/object_ref/image_sha256/phash
├── reference_id/watch_entity_id/content_cluster_id/seller_group_id/acquisition_batch_id
├── angle/source_domain/capture_style
├── label + grade + factory
├── evidence_tier/evidence_record_id
└── rights_record_id
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
propose_source_label(source, *, marketplace=None, listing_text=None) -> Label | None
aggregate_angles(scores_per_angle) -> dict          # simple rule, 0 trainable params (CK-19)
publish_gate(standard_acc, cross_source_acc, max_delta) -> bool
goldset_regression_gate(current_acc, baseline_acc) -> bool
```

**Adapter signatures** (external-system gates):
```python
fit_temperature(logits, labels) -> float            # SKILL.md:157-168, val split only
quality_gate(image) -> Pass | Rejection             # pre-inference, SKILL.md:174-175
authorize_candidate(policy, rights, *, now) -> AcquisitionDecision
collect_batch(records, policy, rights, *, cos_client, bucket, batch_id, now) -> BatchReceipt
adjudicate(candidate, reviews) -> EvidenceRecord
evaluate_readiness(records, reference_id) -> ReadinessReport
authorize_run(credit, usage, pricing, post_promo, spec, events, *, now) -> AuthorizationDecision
launch_training(clients, authorized_run, *, confirm: bool) -> RunEvent  # false refuses (CK-20)
```

### Expected Changes

```
watch-authenticity/
├── core/
│   ├── types.py                     # REWORK: model vocabulary + Prediction/Rejection
│   └── rules.py                     # REWORK: clip, ceiling, candidate proposal, aggregation, gates
│                                     #      goldset_regression_gate (all pure, torch-free)
├── acquisition/{types,source_policy,collect,partner_import}.py  # NEW: rights-gated candidates/receipts
├── scrapers/<source>.py             # REWORK: approved API/feed/export parsing only
├── labeling/{review,adjudicate,manifest}.py # REWORK: evidence-backed accepted rows
├── dataset/{entities,dedup,crop,augment,readiness,splits,freeze,reference_catalog}.py
├── config/{source_policies,reference_catalog}.json
├── adapters/backbone.py             # NEW: frozen DINOv2 loader
├── model/{heads,rejection}.py       # NEW
├── calibration/temperature_scaling.py       # NEW: fit_temperature + ceiling wrap
├── quality_gate/blur_light_check.py # NEW
├── explainability/gradcam.py        # NEW
├── inference/predict.py             # NEW: inference orchestration
├── evaluation/{evaluate,release_gate}.py     # NEW
├── infra/{ibm_types,ibm_evidence,ibm_cost_guard,ibm_cos,ibm_vpc,ibm_train,ibm_watchdog}.py
│                                     # REWORK: IBM VPC L4 + COS, credit/TTL/teardown-gated
├── infra/{bootstrap_host.sh,code-engine-watchdog.yaml} # secretless boot + independent TTL cleanup
├── training/{data,run}.py            # NEW: real immutable training producer
├── Containerfile.train / requirements-train.lock # NEW: digest-pinned GPU runtime
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
- [ ] core/types.py + acquisition/types.py (model and evidence/rights/dataset contracts)
  - Reuse: None — task-specific value objects
- [ ] core/rules.py: clip_confidence, apply_grade4_ceiling, propose_source_label,
      aggregate_angles, publish_gate, goldset_regression_gate (pure, torch-free)
  - Reuse: None — task-specific business rules; implemented ONCE here (CK-22)
- [ ] pyproject/preflight (ruff + mypy + pytest)
  - Reuse: None — no CI convention exists in repo

Phase 2: Evidence-gated dataset acquisition (depends on Phase 1 core and COS Task 3)
- [ ] Execute `04c-data-acquisition.md` Tasks 1–6: contracts, source/rights policy, immutable collection, entity grouping, evidence review and readiness freeze
  - Reuse: `infra.ibm_cos.upload_verified`, ImageHash, existing adapters as parsing-only seams
- [ ] Rework scrapers into approved API/feed/export adapters; no adapter writes final labels
  - Reuse: `acquisition.source_policy`, `core.propose_source_label`
- [ ] Preserve full sanitized image plus hashed per-angle derivatives; classify capture style instead of deleting all studio images
  - Reuse: ImageHash, Pillow/opencv, `dataset.crop`
- [ ] Build grouped standard/cross-source/factory/temporal/gold splits from Tier A/B records only
  - Reuse: `dataset.entities`, `dataset.readiness`, immutable manifest/freezes

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
- [ ] Replace AWS adapter with `infra/ibm_train.py`: VPC profile `gx3-16x80x1l4`, immutable Container Registry image, `confirm=True`, TTL and teardown in `finally`
  - Reuse: existing cost-confirmation boundary; replace provider implementation under AD-2
- [ ] Replace S3 adapter with `infra/ibm_cos.py` for dataset/checkpoint/report sync
  - Reuse: existing upload/download contract; IBM COS SDK
- [ ] Add `infra/ibm_cost_guard.py`: fresh credit/expiry/SKU evidence, conservative run estimate, safety reserve and fail-closed no-overage decision
- [ ] Add `infra/ibm_vpc.py` + `infra/ibm_watchdog.py`: deterministic create reconciliation, no-inbound VSI, Code Engine watchdog and observed-404 deletion proof
- [ ] Wire IAM/trusted profile plus IBM Logs/Monitoring; persist hash-chained run events, image/data lineage, estimated/observed usage and teardown proof
- [ ] Add `training/data.py` + `training/run.py`: replace the fixture-only smoke path with the real immutable training producer before any GPU run
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
Phase 2
  Step 04a-manifest [sonnet] (needs core) ──┬──► 05a-scrapers [sonnet] (needs 04a)
                                             └──► 05c-augment-splits [sonnet] (needs 04a)
  Step 04b-infra [sonnet] (IBM migration rework; needs none)
  Step 05b-dedup-crop [sonnet] (needs none)
  04a/04b/05b start together at Phase-2 start; 05a+05c start once 04a finishes — peak width 3
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
| `01-project-scaffolding` [DONE] | Phase 1 | haiku | haiku | None | `02-core-types` | `.specs/sub-tasks/train-watch-authenticity-model.feature/01-project-scaffolding.md` |
| `02-core-types` [DONE] | Phase 1 | sonnet | sdd:developer | None | `01-project-scaffolding` | `.specs/sub-tasks/train-watch-authenticity-model.feature/02-core-types.md` |
| `03-core-rules` [DONE] | Phase 1 | sonnet | sdd:developer | `02-core-types` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/03-core-rules.md` |
| `04a-manifest` [DONE] | Phase 2 | sonnet | sdd:developer | `03-core-rules` | `04b-infra` | `.specs/sub-tasks/train-watch-authenticity-model.feature/04a-manifest.md` |
| `04b-infra` [REWORK REQUIRED — AWS→IBM] | Phase 2 | sonnet | sdd:developer | None | `04a-manifest` | `.specs/sub-tasks/train-watch-authenticity-model.feature/04b-infra.md` |
| `04c-data-acquisition` [PLAN READY — IMPLEMENTATION REQUIRED] | Phase 2 | sonnet | sdd:developer | `04b-infra` Task 3 | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/04c-data-acquisition.md` |
| `05a-scrapers` [DONE] | Phase 2 | sonnet | sdd:developer | `04a-manifest` | `05b-dedup-crop`, `05c-augment-splits` | `.specs/sub-tasks/train-watch-authenticity-model.feature/05a-scrapers.md` |
| `05b-dedup-crop` [DONE] | Phase 2 | sonnet | sdd:developer | None | `04a-manifest`, `04b-infra`, `05a-scrapers`, `05c-augment-splits` | `.specs/sub-tasks/train-watch-authenticity-model.feature/05b-dedup-crop.md` |
| `05c-augment-splits` [DONE] | Phase 2 | sonnet | sdd:developer | `04a-manifest` | `05a-scrapers`, `05b-dedup-crop` | `.specs/sub-tasks/train-watch-authenticity-model.feature/05c-augment-splits.md` |
| `06-backbone-heads` [DONE] | Phase 3 | sonnet | sdd:developer | `05c-augment-splits` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/06-backbone-heads.md` |
| `07a-rejection` [DONE] | Phase 3 | sonnet | sdd:developer | `06-backbone-heads` | `07b-calibration` | `.specs/sub-tasks/train-watch-authenticity-model.feature/07a-rejection.md` |
| `07b-calibration` [DONE] | Phase 3 | sonnet | sdd:developer | `06-backbone-heads` | `07a-rejection` | `.specs/sub-tasks/train-watch-authenticity-model.feature/07b-calibration.md` |
| `08a-quality-gate` [DONE] | Phase 4 | sonnet | sdd:developer | `03-core-rules` | `08b-gradcam` | `.specs/sub-tasks/train-watch-authenticity-model.feature/08a-quality-gate.md` |
| `08b-gradcam` [DONE] | Phase 4 | sonnet | sdd:developer | `06-backbone-heads` | `08a-quality-gate` | `.specs/sub-tasks/train-watch-authenticity-model.feature/08b-gradcam.md` |
| `09-inference-predict` | Phase 4 | sonnet | sdd:developer | `06-backbone-heads`, `07a-rejection`, `07b-calibration`, `08a-quality-gate`, `08b-gradcam` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/09-inference-predict.md` |
| `10-evaluate` | Phase 5 | sonnet | sdd:developer | `09-inference-predict` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/10-evaluate.md` |
| `11-release-gate` | Phase 5 | sonnet | sdd:developer | `10-evaluate` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/11-release-gate.md` |
| `12-publish-weights` | Phase 5 | sonnet | sdd:developer | `11-release-gate` | None | `.specs/sub-tasks/train-watch-authenticity-model.feature/12-publish-weights.md` |

**Resume order after the 05/11/2026 infrastructure decision:**

1. Implement `04b-infra` Tasks 1–3 through verified COS; no GPU provisioning.
2. Execute `04c-data-acquisition` Tasks 1–6 and repeat the Phase 1/2 reviews because source-label and manifest contracts change.
3. Implement inference plus `04c` Task 7, then execute evaluation/release against one frozen dataset hash.
4. Before a real acquisition/training run, prove source rights and IBM promo balance/expiry/SKU/cost; both external paths remain `PENDENTE E2E`.

### Phase Overview

#### Phase 1: Foundation (pure core + preflight) [REVIEWED; CONTRACT REWORK PLANNED]

Steps: `01-project-scaffolding`, `02-core-types`, `03-core-rules`
Reviewer model: `opus`
Acceptance Criteria that should be fulfiled:
Checklist items:
- `CK-1` — Does every reported confidence fall strictly within (0.1%, 99.9%)? (`clip_confidence` defined + tested)
- `CK-3` — Is genuine-confidence capped at ~60% for grade-4-consistent output? (`apply_grade4_ceiling` defined + tested)
- `CK-8` — Is there a rule that blocks publish on cross-source degradation? (`publish_gate` predicate)
- `CK-12` — Does the gold-set regression gate block publishing a regressed version? (`goldset_regression_gate` predicate)
- `CK-13` — Do OLX/ML "original" claims return no proposed label and remain unable to build a training record?
- `CK-18` — Does any part treat an "original"-labeled OLX/ML ad as genuine? (pitfall — must be NO)
- `CK-19` — Does aggregation use a learned model instead of the simple rule? (pitfall — `aggregate_angles` has 0 trainable params)
- `CK-22` — Is the code free of duplicated grade-map/clamp/gate logic? (single-source `core/rules.py`)
- `CK-27` — Does the rights gate refuse r/rolex before download without both required permissions?
- `CK-29` — Does any part include Reddit content without a permission record? (pitfall — must be NO)

Rubrics:
- `Confidence Bound & Calibration Integrity` (clamp portion)
- `Grade-4 Superclone Ceiling Enforcement` (ceiling function)

#### Phase 2: Evidence-gated dataset acquisition (+ training infra) [REVIEW INVALIDATED — IBM + EVIDENCE REWORK]

Steps: historical prototypes `04a/05a/05b/05c`, IBM `04b`, canonical rework `04c-data-acquisition`
Reviewer model: `opus`
Acceptance Criteria that should be fulfiled:
Checklist items:
- `CK-10` — Does every training photo have candidate/rights/evidence/entity/hash lineage?
- `CK-11` — Are duplicates clustered, uncertain catalog copies excluded and confirmed studio photos retained correctly?
- `CK-13` — Are OLX/ML "original" listings excluded at ingestion?
- `CK-16` — Does training apply lighting/blur/JPEG augmentation?
- `CK-18` — Does ingestion treat an "original" OLX/ML ad as genuine? (pitfall — must be NO)
- `CK-20` — Does training-launch tooling require an explicit cost-confirmation step?
- `CK-30` — Is the operational path IBM-native with credit/TTL/teardown protection and no paid fallback?
- `CK-21` — Does a crop/region-extraction step isolate each guided angle?
- `CK-27` — Does source governance exclude r/rolex before download without permissions?
- `CK-28` — Does Mercado Livre sourcing route through the official Developer API, not raw scraping?
- `CK-29` — Does any scraper include Reddit content without a permission record? (pitfall — must be NO)
- `CK-31/32` — Are current rights plus Tier A/B evidence mandatory, with no source-as-truth path?
- `CK-33/34` — Are leakage groups indivisible and quantitative readiness gates enforced?
- `CK-35/36` — Do revocation and new-reference admission fail closed?

Rubrics:
- `Dataset Manifest Completeness & Auditability`

#### Phase 3: Model + calibration [REVIEWED]

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
