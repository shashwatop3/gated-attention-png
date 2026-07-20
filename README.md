# Gating Mechanisms in Attention-Based Neural Networks

**Empirical evaluation on GPT-2 and BERT with a novel Patch-Norm Gate (PNG)**

Vedh Adla, Shashwat Chaturvedi, Vasanth Prabhu Kumble
Department of Information Technology, National Institute of Technology Karnataka, Surathkal
B.Tech (AI) Project, 2025–2026

Full report: [`report/gating_mechanisms_attention_networks_report.pdf`](report/gating_mechanisms_attention_networks_report.pdf)

## Overview

Gating mechanisms — from LSTMs and Highway Networks to SwiGLU and modern
state-space/linear-attention models — are a foundational tool in neural
network design. [Qiu et al. (2025)](https://arxiv.org/abs/2505.06708) ran the
most comprehensive study to date of gating variants in attention, on
large-scale (1.7B–15B parameter) MoE and dense LLMs, and found that a
**head-specific sigmoid gate applied after the SDPA output (G1)** reliably
improves perplexity and eliminates the attention-sink artifact.

This project asks: **do those findings hold at a much smaller, task-fine-tuned
scale?** We re-implement five gate positions (G1–G5) from the base paper via
zero-architecture-change forward hooks, and introduce a new variant:

### Patch-Norm Gate (PNG)

PNG augments G1 with a token-norm correction term, so the gate bias adapts
dynamically to how large a token's hidden-state norm is — directly and
cheaply suppressing attention-sink tokens (`[CLS]`/`BOS`) without any
explicit regularization:

```
G_i = sigmoid( h_i @ W_theta  -  beta * ||h_i||_2 * e )
```

where `h_i` is the pre-norm hidden state at position `i`, `W_theta` and `e`
(per-head bias vector) are jointly learned, and `beta = 0.1` is a fixed decay
coefficient.

## Results

**GPT-2 small on WikiText-103** (language modeling, full fine-tuning, 3 epochs):

| Variant | Test PPL | ΔPPL vs. baseline | Attention sink |
|---|---|---|---|
| Baseline | 24.15 | — | 0.393 |
| G1 – SDPA output | 10.07 | 14.07↓ | 0.141 |
| G2 – Value proj | 10.07 | 14.08↓ | 0.143 |
| G3 – Key proj | 11.41 | 12.74↓ | 0.235 |
| G4 – Query proj | 10.97 | 13.17↓ | 0.245 |
| G5 – Dense output | 10.22 | 13.93↓ | 0.207 |
| **PNG (ours)** | **8.78** | **15.37↓** | **0.098** |

**BERT-base on HateXplain** (3-class hate speech detection, full fine-tuning, 5 epochs):

| Variant | Accuracy | Macro-F1 | Precision | Recall |
|---|---|---|---|---|
| Baseline | 67.31% | 0.6657 | 0.6650 | 0.6687 |
| G1 – SDPA output | 68.45% | 0.6659 | 0.6666 | 0.6718 |
| G5 – Final output | 68.66% | 0.6700 | 0.6700 | 0.6755 |
| **PNG (ours)** | **68.81%** | **0.6761** | **0.6788** | **0.6786** |

**Key findings**

- Gating consistently improves both language modeling and classification over
  an ungated baseline, across two different model families (decoder-only
  GPT-2, encoder-only BERT) and tasks.
- The position ordering **G1 ≈ G2 > G5 > G4 > G3** from the base paper holds
  at 10× smaller scale.
- Query-dependent, post-SDPA gating (G1, PNG) is consistently the most
  effective at eliminating attention sink; PNG achieves the lowest sink score
  in every layer of both architectures.
- PNG's gate score correlates negatively with token norm (Pearson `r ≈ -0.38`
  on BERT), confirming the mechanism suppresses high-norm sink tokens
  (`[CLS]`, punctuation) as designed.
- PNG achieves the best token-attention/human-rationale alignment on
  HateXplain (`r = 0.087` vs. baseline `r = 0.056`, a 56% relative
  improvement).

See the [report](report/gating_mechanisms_attention_networks_report.pdf) for
the full ablation (all 7 variants × both tasks), attention-sink heatmaps,
gate-sparsity analysis, and per-class breakdowns.

## Repository structure

```
notebooks/
  01_gpt2_g1_g5_training.ipynb        GPT-2 small fine-tuning, gate positions G1-G5, WikiText-103
  02_gpt2_png_training.ipynb          + PNG gate training (loads G1-G5 checkpoints)
  03_gpt2_evaluation_figures.ipynb    Perplexity, attention-sink, gate-sparsity analysis, all GPT-2 figures
  04_bert_hatexplain_training_eval.ipynb   BERT-base fine-tuning + evaluation, all 7 variants, HateXplain

report/
  gating_mechanisms_attention_networks_report.pdf   Full write-up (NITK B.Tech project report)

archive/
  Earlier project iterations (ViT/ImageNet/ADE20K, DistilBERT/TweetEval, a
  frozen-probe BERT ablation) that predate the final scope. See archive/README.md.
```

## Gate positions (G1–G5)

All five positions modulate a tensor within self-attention via a
zero-initialized sigmoid gate (`Y' = Y ⊙ σ(X W_theta)`), injected through
PyTorch forward hooks on `BertSelfAttention` / `GPT2Attention` — **no
architecture changes to the pretrained model are required**.

| Gate | Modulates | Position |
|---|---|---|
| G1 | SDPA output (per head) | After SDPA |
| G2 | Value vectors | Before SDPA |
| G3 | Key vectors | Before SDPA |
| G4 | Query vectors | Before SDPA |
| G5 | Concatenated context | Post-reshape |
| PNG | SDPA output, norm-corrected | After SDPA |

## Reproducing

```bash
pip install -e .
jupyter lab
```

Notebooks were developed and trained on Kaggle/Colab GPUs; training cells
expect a CUDA device (they will fall back to CPU/MPS but at prohibitive
speed for a full run). `03_gpt2_evaluation_figures.ipynb` expects trained
checkpoints from notebooks 01/02 in a `checkpoints_llm_v3/` or
`checkpoints_llm_unfrozen/` directory — checkpoints are not tracked in this
repo (see `.gitignore`) since they exceed reasonable git storage; re-run 01
and 02 to regenerate them, or point the checkpoint search path at your own.

## Reference

This work extends:

> Z. Qiu et al., "Gated attention for large language models: Non-linearity,
> sparsity, and attention-sink-free," arXiv:2505.06708, 2025.
> https://arxiv.org/abs/2505.06708

## License

[MIT](LICENSE)
