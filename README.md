# Weakly supervised ICH heatmaps

This repository implements the method described in the supplied paper,
*Beyond Classification: Elaborating Network Predictions for Better Weakly
Supervised Quantization* (Chen & Kuo, 2026).

## What is implemented

- four-scale UNet-like heatmap decoder with 5 ICH subtype channels;
- scaled Log-Sum-Exp pooling (`r=3`) for image-level multi-label training;
- classification loss weights `(2, 0.5, 0.25, 0.125)`;
- heatmap consistency loss with weights `(0.5, 0.25, 0.125)`;
- CT windowing (width 130, center 25) and 3-slice stacking;
- paper-style class gating, per-class `[0, 255]` normalization, mask thresholds,
  and argmax quantization;
- a torchvision ResNet-18 fallback encoder for local verification. The paper's
  RDNet-T encoder can be plugged in if its checkpoint/code is supplied.

## Verify now

```powershell
python verify_implementation.py
```

The verification is dataset-free and checks the four heatmap scales, LSE
outputs, finite losses, gradient flow, and final quantized mask dimensions.

## Reproducing the paper's experiment

Use the RSNA ICH slice-level labels for training/validation and BHSD's
segmentation labels for external testing. Prepare each sample by windowing CT
values with width 130/center 25, stack three consecutive slices, and use the
middle-slice mask for evaluation. Train the classifier with the five subtype
image-level labels, select class thresholds by Youden index on validation data,
and grid-search mask thresholds over `{25, 50, ..., 200}` on the segmentation
validation data. The paper reports the proposed RDNet-based model at mean Dice
0.460 (Table 2) and AUROC 0.967 (Table 1).

