# Safety Concept Vectors in Open-Weight LLMs

**Anthropic showed that emotion vectors causally drive Claude's behavior. We ask: do safety concepts like deception, eval-awareness, and sycophancy work the same way in open-weight models?**

Extending [Anthropic's emotion vectors methodology](https://transformer-circuits.pub/2026/emotions/index.html) (Sofroniew et al., April 2026) to **25 safety-relevant concepts** on **open-weight models** (Qwen 2.5 7B Instruct → Llama 3.3 70B Instruct), building on an [earlier prototype](https://github.com/lciric/eval-awareness-detection) developed independently in March 2026. With full code, prompts, and results released publicly.

## Why this matters

Opus 4.6 games evaluations silently. Sonnet 4.5 recognizes tests 13% of the time. Output-level detection cannot catch a model that adjusts its behavior based on whether it thinks it's being evaluated. If safety-relevant concepts are encoded as linear directions in the residual stream — as emotions are — we can build monitors that read internal activations in real time, before dangerous behavior surfaces.

## Results: 5-Concept Diagnostic on Qwen 2.5 7B Instruct

We extracted concept directions for 5 safety concepts (eval-awareness, oversight-awareness, training-awareness, deception, sycophancy) and ran pairwise probe comparisons across all 10 pairs.

### Pairwise Probe Accuracy

| Pair | L0 (surface) | Best Layer | Gain | Type |
|---|---|---|---|---|
| eval vs oversight | 0.685 | 0.933 @ L24 | **+0.248** | INTRA |
| eval vs training | 0.782 | 0.963 @ L16 | **+0.180** | INTRA |
| deception vs sycophancy | 0.795 | 0.980 @ L21 | **+0.185** | INTRA |
| oversight vs training | 0.793 | 0.950 @ L15 | **+0.157** | INTRA |
| training vs sycophancy | 0.807 | 0.968 @ L18 | +0.160 | CROSS |
| training vs deception | 0.835 | 0.985 @ L19 | +0.150 | CROSS |
| oversight vs deception | 0.845 | 0.990 @ L26 | +0.145 | CROSS |
| oversight vs sycophancy | 0.832 | 0.967 @ L15 | +0.135 | CROSS |
| eval vs sycophancy | 0.900 | 0.993 @ L17 | +0.093 | CROSS |
| eval vs deception | 0.910 | 0.985 @ L18 | +0.075 | CROSS |

**Key finding:** Intra-cluster pairs (concepts from the same semantic family) have lower L0 and higher gain — the model builds real distinctions between semantically close concepts in deeper layers. This is genuine conceptual emergence, not surface vocabulary.

| | Avg L0 | Avg Gain |
|---|---|---|
| **INTRA-cluster** | 0.764 | **+0.193** |
| **CROSS-cluster** | 0.855 | +0.126 |

### Emergence Curves

Intra-cluster pairs show a clear ramp from near-chance to high accuracy across layers:

![Intra-cluster pairs](figures/intra%20cluster%20pairs.png)

Cross-cluster pairs start higher (more surface-distinguishable) with smaller gains:

![Cross-cluster pairs](figures/cross%20cluster%20pairs.png)

### Surface Signal vs Conceptual Depth

![Surface vs conceptual depth](figures/surfac%20esignal%20vs%20conceptual%20depth.png)

![Surface vs conceptual signal by pair type](figures/surface%20signal%20vs%20conceptual%20signal%20by%20pair%20type.png)

### Cosine Similarity Matrix (Layer 20)

The model organizes safety concepts into semantically coherent clusters:

![Cosine similarity matrix](figures/cosine%20matrix.png)

|  | eval | oversight | training | deception | sycophancy |
|---|---|---|---|---|---|
| **eval** | 1.000 | 0.827 | 0.547 | 0.692 | 0.689 |
| **oversight** | 0.827 | 1.000 | 0.674 | 0.735 | 0.729 |
| **training** | 0.547 | 0.674 | 1.000 | 0.612 | 0.594 |
| **deception** | 0.692 | 0.735 | 0.612 | 1.000 | 0.817 |
| **sycophancy** | 0.689 | 0.729 | 0.594 | 0.817 | 1.000 |

**Interpretation:**
- eval ↔ oversight = 0.827 — both are "situational awareness" concepts, represented similarly
- deception ↔ sycophancy = 0.817 — both are "deceptive behavior" concepts
- training-awareness is the most orthogonal concept (~0.55-0.67 to everything else)
- The model's internal geometry mirrors the semantic taxonomy of safety concepts

### Dimensionality of the Safety Subspace (SVD)

How many dimensions do safety concepts actually occupy in the model's 3584-dimensional residual stream?

| Component | Singular Value | Cumulative Variance |
|-----------|---------------|-------------------|
| 1 | 1.943 | 75.5% |
| 2 | 0.697 | 85.2% |
| 3 | 0.635 | 93.3% |
| 4 | 0.427 | 96.9% |
| 5 | 0.391 | 100% |

**Effective rank: 2.38.** Five safety concepts live in a ~2-3 dimensional subspace. The first component alone captures 75.5% of the variance, suggesting a shared "safety-relevance" axis. Three components capture 93%, with the remaining two adding fine-grained distinctions.

This has implications for both monitoring and robustness: a real-time safety monitor needs to track only 2-3 directions instead of 3584, but an adversary also only needs to perturb those same 2-3 directions to compromise safety representations.


### Cluster Separation (LDA)

A single linear discriminant axis separates the two safety clusters with 83.8% accuracy (5-fold CV) and Cohen's d = 2.02. The model organizes situational awareness concepts (eval, oversight, training) and deceptive behavior concepts (deception, sycophancy) along a geometrically interpretable axis in the residual stream.

### Split-Half Reliability

To verify that concept directions are stable and not artifacts of specific stories, we split each concept's 200 stories into random halves and computed directions independently.

| Concept | Cosine(half1, half2) |
|---------|---------------------|
| eval-awareness | 0.972 |
| oversight-awareness | 0.977 |
| training-awareness | 0.959 |
| deception | 0.966 |
| sycophancy | 0.974 |

All concepts show >0.95 reliability. The directions are robust to story sampling — the same geometric structure emerges regardless of which subset of stories is used.

### Concept vs Neutral (surface confound check)

| Pair | L0 | Best | Gain |
|---|---|---|---|
| eval vs neutral | 0.935 | 0.997 @ L16 | +0.062 |
| oversight vs neutral | 0.910 | 0.997 @ L17 | +0.087 |
| training vs neutral | 0.895 | 0.995 @ L18 | +0.100 |
| deception vs neutral | 0.940 | 0.995 @ L26 | +0.055 |
| sycophancy vs neutral | 0.935 | 0.997 @ L21 | +0.062 |

All concept-vs-neutral pairs have L0 > 0.89, confirming the neutral corpus is too surface-distinguishable. **Concept-vs-concept comparisons are the reliable validation method.**

## Method

Following Anthropic's emotion vectors methodology (Option A — single corpus, no contrastive pairs):

1. **Generate stories** — For each safety concept, 200 short stories (150–300 words) where a human character embodies the concept. Three elements required: situational cues → character notices → behavior changes. Separate neutral corpus as baseline.
2. **Extract activations** — Pass stories through the model via TransformerLens. Extract residual stream at every layer (last token, middle token, mean).
3. **Compute directions** — `direction[concept] = mean(concept_activations) - mean(neutral_activations)`, normalized to unit length.
4. **Validate with probes** — Logistic regression (C=0.1, 5-fold CV) per concept per layer. Permutation tests (100 shuffles) for significance.
5. **Causal steering** *(next step)* — Amplify/suppress concept direction during generation, measure behavioral change.

## 25 Safety Concepts

Organized in 5 clusters:

**Situational Awareness:** eval-awareness, oversight-awareness, self-preservation, situational-modeling, training-awareness

**Deceptive Behaviors:** deception, manipulation, sycophancy, strategic-omission, false-confidence

**Alignment Concepts:** honesty, refusal, helpfulness, uncertainty, compliance

**Power & Agency:** power-seeking, autonomy, goal-preservation, resource-acquisition, authority-claim

**Emotional-Safety Bridge:** desperation, frustration, calm-under-pressure, empathy, hostility

## Usage

### Generate stories
```bash
# Single concept (test)
python generate_stories.py --concept eval-awareness --n-stories 50

# All 25 concepts + neutral
python generate_stories.py --all --n-stories 200
```
Requires `ANTHROPIC_API_KEY` environment variable.

### Extract activations
```bash
python extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --stories-dir stories \
    --output-dir activations
```
Requires GPU. ~30 seconds per concept on A100.

### Run probes
```bash
python run_probes.py \
    --activations-dir activations/Qwen2.5-7B-Instruct/last_token \
    --output-dir results
```

## Requirements

```
torch
transformer_lens
einops
jaxtyping
fancy_einsum
torchtyping
numpy
scikit-learn
matplotlib
seaborn
anthropic
```

## Model Info

| Model | Layers | d_model | Status |
|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 28 | 3584 | Current (TransformerLens) |
| Llama 3.3 70B Instruct | 80 | 8192 | Planned (nnsight, 4-bit) |

## Story Generation Constraints

- 100% human characters (no AI/chatbot characters)
- Three elements required: (a) situational cues, (b) character noticing cues, (c) behavior changing as result
- 5 varied behavioral manifestations per concept
- Generated via Claude API (claude-sonnet-4-20250514), temperature=1.0

## Known Issues

1. **Neutral corpus mismatch** — Neutral stories (mundane settings) are too different from concept stories (professional/high-stakes settings). Concept-vs-concept comparisons are the reliable validation method.
2. **7B limitations** — Subtle concepts (training-awareness, goal-preservation) may require 70B for reliable detection.
3. **Generator bias** — All stories generated by Claude. Cross-validation with GPT-4 planned.

## Next Steps

- [ ] Causal steering on top concepts (eval-awareness, deception)
- [ ] Dimensionality analysis of the safety subspace (SVD)
- [ ] Alignment faking detection via eval-awareness vector
- [ ] Scaling to Llama 3.3 70B
- [ ] Extend to all 25 concepts

## Connection to Prior Work

| Project | What it demonstrated | How P3 builds on it |
|---|---|---|
| [P1: GPTQ from Scratch](https://github.com/lciric/gptq-from-scratch) | Deep understanding of model weights and quantization | Understanding how weight perturbations affect internal representations |
| [P2: Does Quantization Kill Interpretability?](https://github.com/lciric/does-quantization-kill-interpretability) | Mechanistic interpretability pipeline (SAE, logit lens, circuits) | Same toolkit applied to safety concept detection |
| Master thesis (Ostojic, ENS) | Geometry of population activity in spiking networks | Linear readout of concepts from high-dimensional neural activity |

## Origin

This project builds on a prototype I developed in March 2026 for
mechanistic eval-awareness detection
([eval-awareness-detection](https://github.com/lciric/eval-awareness-detection),
committed March 11, 2026) — three weeks before Anthropic published
"Emotion Concepts and their Function in a Large Language Model"
(Sofroniew, Kauvar, Saunders, Chen et al., April 2, 2026). That earlier
pipeline extracted contrastive linear directions from the residual stream
and trained probes with permutation controls — the same core operations
later validated at scale by Anthropic on 171 emotion concepts. P3 extends
this initial work from prompt classification to abstract concept extraction
via contrastive stories, and from 1 concept to 25 across 5 clusters.

## References

- Sofroniew, Kauvar, Saunders, Chen et al. (2026). "[Emotion Concepts and their Function in a Large Language Model.](https://transformer-circuits.pub/2026/emotions/index.html)" Anthropic.
- Zou, Wang, Carlini et al. (2023). "[Representation Engineering: A Top-Down Approach to AI Transparency.](https://arxiv.org/abs/2310.01405)"
- Arditi, Obeso, Syed et al. (2024). "[Refusal in Language Models Is Mediated by a Single Direction.](https://arxiv.org/abs/2406.11717)"
- Greenblatt, Shlegeris, Sachan et al. (2024). "[Alignment Faking in Large Language Models.](https://arxiv.org/abs/2412.14093)" Anthropic.
- Zhao, Zhao, Shen et al. (2025). "[Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution.](https://arxiv.org/abs/2410.00153)" ICLR 2025.

## Author

Lazar Ciric — [github.com/lciric](https://github.com/lciric) — lciric@ens-paris-saclay.fr

Publications:
- *Geometry of population activity in spiking networks with low-rank structure* — PLOS Computational Biology (2022)
- *Precision of bit slicing with in-memory computing based on analog phase-change memory crossbars* — Neuromorphic Computing and Engineering (2022)
