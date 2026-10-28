"""Verify the local environment and official model entry point.

This is intentionally independent of the RSNA/BHSD datasets.  It validates
Stages 1-2 only: imports, dependency visibility, model construction, and an
optional 512x512 forward pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path

import torch


REQUIRED = {
    "timm": "timm",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "yaml": "PyYAML",
    "albumentations": "albumentations",
    "pydicom": "pydicom",
    "nibabel": "nibabel",
    "torchmetrics": "torchmetrics",
    "pytorch_grad_cam": "grad-cam",
    "torchsummary": "torchsummary",
    "absl": "absl-py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", action="store_true", help="run a 512x512 CPU forward pass")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo))

    print("python:", sys.version.split()[0])
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))

    missing = [pip_name for module, pip_name in REQUIRED.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("MISSING PACKAGES:", ", ".join(missing))
        raise SystemExit(2)
    print("dependency check: PASS")

    from models.model_segrdnet import SegRDNet

    model = SegRDNet(num_classes=6, init_checkpoint=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("model construction: PASS")
    print("parameters:", parameter_count)

    if args.forward:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x = torch.zeros(1, 3, 512, 512, device=device)
        with torch.no_grad():
            output, consistency, intermediate = model(x, eval_mode=False)
        assert output.shape == (1, 6), output.shape
        assert len(consistency) == 3, len(consistency)
        assert len(intermediate) == 4, len(intermediate)
        assert all(torch.isfinite(value).all() for value in (output, *consistency, *intermediate))
        print("forward pass: PASS")
        print("output:", tuple(output.shape))
        print("intermediate:", [tuple(value.shape) for value in intermediate])
        print("consistency losses:", [float(value) for value in consistency])


if __name__ == "__main__":
    main()
