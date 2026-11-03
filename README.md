# Small Models and Attention

Four self-contained deep learning experiments exploring how small models behave under pressure: how confident they get, how they generate sequences, how attention controls what information flows where, and how much a vision model leans on positional cues. Built for **Deep Learning** (Prof. Nipun Batra, IIT Gandhinagar, Assignment 2, September 2026).

## Overview

| # | Experiment | Core Idea |
|---|------------|-----------|
| Q1 | Label Smoothing | Hard vs. soft targets on FashionMNIST, and their effect on confidence calibration |
| Q2 | Character-Level Name Generator | A tiny embedding-based MLP that learns to spell names letter by letter |
| Q3 | Causal Self-Attention | Manually implemented attention masking, verified to block future-token leakage |
| Q4 | Vision Transformer | A minimal ViT from scratch, tested for its reliance on learned position embeddings |

## Q1: Label Smoothing and Confidence Calibration

Two identical MLP classifiers (`784 -> 128 -> ReLU -> 10`) were trained from the same initial weights on 6,000 FashionMNIST images, one with standard cross-entropy (hard targets) and one with label smoothing of 0.1 (soft targets).

**Results (best validation epoch, epoch 8 for both):**

| Model | Test Accuracy | Mean Confidence |
|-------|---------------|-----------------|
| Hard targets | 82.26% | 0.807 |
| Smoothed (α=0.1) | 82.48% | 0.715 |

Accuracy barely moved, but the confidence histogram tells the real story: the hard model put 3,830 of 10,000 test images in the 0.95-1.0 confidence bin, versus only 1,200 for the smoothed model, whose mass shifted into the 0.90-0.95 bin instead. With one-hot targets, cross-entropy keeps rewarding higher confidence all the way to 1.0, which pushes logits apart indefinitely. With a soft target (0.91 on the true class), the loss is minimized at 0.91 confidence, so there is no pressure to go further, and the model ends up noticeably less overconfident for the same accuracy.

## Q2: Character-Level Name Generator

A next-character predictor built from a learned 2-dimensional character embedding table feeding a small MLP (`6 -> 32 -> ReLU -> 27`), trained on a context window of 3 characters over 6,478 names (5,182 train, 647 validation, 649 test).

- **Selected epoch:** 20 (out of 20), best validation loss 2.2883 nats/token
- **Test loss:** 2.3217 nats per predicted token
- Greedy decoding always produces the same name (`sara`), since the same fixed start context deterministically picks the same argmax character every time
- Sampling at low temperature (0.5) stays close to training patterns and produces short, plausible names (`ran`, `bach`, `paj`); temperature 2 flattens the distribution and produces long, unlikely strings (`roinbachcvmprach`)

## Q3: Causal Self-Attention From Scratch

A toy 3-token, 2-dimensional attention block (identity projection matrices for Q, K, V, and output) was used to directly verify what a causal mask does to attention weights.

- Without masking, every query attends to all 3 positions, including the future ones (row 0 gives weight 0.40 to position 2)
- With a causal mask, all 3 above-diagonal entries are exactly 0.0, and each row renormalizes only over visible positions
- Changing the value at position 2 (the last token) shifts the first two output rows by **2.84** without a mask, and by exactly **0.0** with the causal mask applied, directly proving zero information leakage from future to past positions

This matters for next-token prediction: if position *i* could see token *i+1* through attention, it could simply copy the answer instead of learning to predict it.

## Q4: Vision Transformer and Positional Dependence

A minimal ViT (patch projection `16 -> 32`, a CLS token, a 2-head attention block, and a `32 -> 64 -> 32` MLP, 11,114 parameters total) was trained in two variants: with and without a learned position embedding table, on 49 patches per 28x28 FashionMNIST image.

| Model | Original Test Acc. | Shuffled-Patch Test Acc. | Predictions Unchanged |
|-------|--------------------|--------------------------|------------------------|
| With positions | 76.73% | 17.98% | 18.61% |
| Without positions | 60.93% | 60.93% | 100.00% |

The model with position embeddings learns *where* things are (sleeves at the sides, soles at the bottom) and scores higher on ordered images, but collapses almost entirely once patches are randomly permuted. The model without position embeddings cannot tell patch order apart in the first place, so it scores lower overall but is completely invariant to shuffling, since its CLS token's attention output is a permutation-invariant weighted sum over patch content alone.

## Repository Structure

```
small-models-and-attention/
├── a2_<rollno>.ipynb      # Full notebook: Q1-Q4, code, outputs, explanations
└── README.md
```

## Tech Stack

PyTorch, torchvision (FashionMNIST), NumPy, Matplotlib.

## Notes

All experiments use fixed seeds (42, 43, 44) for data splitting, training, sampling, and patch shuffling, so results are reproducible across runs. This was a group assignment; implementation, experiments, and write-ups were completed collaboratively, with AI tools used only to check understanding and assist with plotting code.
