"""Create three-slice CT PNG inputs from a validated RSNA manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from pydicom import dcmread


def read_pixels(path: str | Path) -> np.ndarray:
    ds = dcmread(path)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return ds.pixel_array.astype(np.float32) * slope + intercept


def window(image: np.ndarray, width: float = 130, center: float = 25) -> np.ndarray:
    low, high = center - width / 2, center + width / 2
    return (np.clip(image, low, high) - low) * (255.0 / (high - low))


def preprocess_study(study: pd.DataFrame, output_dir: str | Path,
                     width: float = 130, center: float = 25,
                     resize: Optional[int] = None) -> pd.DataFrame:
    if study.empty:
        raise ValueError("study is empty")
    study = study.sort_values(["order_value", "image_id"]).reset_index(drop=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pixels = [window(read_pixels(path), width, center) for path in study["dicom_path"]]
    rows = []
    for index, row in study.iterrows():
        image = np.stack([
            pixels[max(index - 1, 0)], pixels[index],
            pixels[min(index + 1, len(pixels) - 1)],
        ], axis=-1).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(image, mode="RGB")
        if resize is not None:
            result = result.resize((resize, resize), Image.Resampling.BILINEAR)
        output_path = output_dir / f"{row.image_id}.png"
        result.save(output_path)
        rows.append({
            "image_id": row.image_id, "png_path": str(output_path.resolve()),
            "study_key": row.study_key, "slice_index": index,
            "IPH": int(row.IPH), "IVH": int(row.IVH), "SAH": int(row.SAH),
            "SDH": int(row.SDH), "EDH": int(row.EDH),
        })
    return pd.DataFrame(rows)


def preprocess_manifest(manifest_csv: str | Path, output_dir: str | Path,
                        output_manifest: str | Path, limit_studies: Optional[int] = None,
                        resize: Optional[int] = 512) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    study_keys = manifest["study_key"].drop_duplicates().tolist()
    if limit_studies is not None:
        study_keys = study_keys[:limit_studies]
    results = [preprocess_study(
        manifest[manifest["study_key"] == study_key], output_dir, resize=resize
    ) for study_key in study_keys]
    if not results:
        raise ValueError("manifest contains no studies")
    output = pd.concat(results, ignore_index=True)
    Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_manifest, index=False)
    print(f"wrote {len(output)} PNG rows to {output_manifest}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--limit-studies", type=int)
    parser.add_argument("--resize", type=int, default=512)
    args = parser.parse_args()
    preprocess_manifest(**vars(args))


if __name__ == "__main__":
    main()
