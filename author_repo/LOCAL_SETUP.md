# Local setup and verification

The official code was written for the authors' environment and assumes a
Linux working directory, CUDA, and an optional pretrained checkpoint. The
repository now has a local dependency manifest and a dataset-free verifier.

## Install

Install a compatible PyTorch build first. Then, from this directory:

```powershell
python -m pip install -r requirements-cpu.txt
```

## Verify Stage 1 and Stage 2

From `author_repo`:

```powershell
python verify_stage_1_2.py
python verify_stage_1_2.py --forward
```

The first command checks imports and model construction. The second also runs
the official SegRDNet at 512x512 and checks output shapes, finite values, and
consistency losses. It does not require RSNA/BHSD data or a checkpoint.

The RDNet tiny factory was corrected so `pretrained=False` is respected. This
is necessary for CPU/offline construction; otherwise the model attempts to
download ImageNet weights even when the caller disables pretraining.
