# Archive — Earlier Project Iterations

This project went through several pivots before converging on the final study
described in [`report/`](../report) and implemented in [`notebooks/`](../notebooks).
These earlier notebooks are kept for history but are **not** part of the final
results — none of the numbers here appear in the report.

## `vit_imagenet_ade20k/`
The original direction: PNG (then "Patch-Norm Gate" in the literal sense — image
patches) applied to `vit_base_patch16_224` / DeiT-B, evaluated on ImageNet-1K
classification and ADE20K segmentation (linear probe), benchmarked against
Darcet et al.'s ViT+Registers. Superseded when the project moved to text
transformers, where PNG was reinterpreted as a **token**-norm gate.
`png_deit_base.pth` is an old checkpoint from this phase (not needed to
reproduce the final report).

## `tweeteval_offensive_language/`
Second iteration: PNG adapted to DistilBERT and BERT-base for binary offensive
language detection on TweetEval/OffensEval (SemEval 2019 Task 6). Superseded
by the HateXplain 3-class setup in the final report, which gave richer
per-class analysis and rationale-alignment evaluation.

## `hatexplain_frozen_probe/`
A frozen-backbone (linear-probe) variant of the final BERT/HateXplain
experiment — only gate parameters and the classifier head were trained. The
report's BERT results use the **unfrozen**, fully fine-tuned setting
(`notebooks/04_bert_hatexplain_training_eval.ipynb`); this probe variant was
an ablation that didn't make it into the final writeup.

## `execution_plan.docx`
Original project planning document.
