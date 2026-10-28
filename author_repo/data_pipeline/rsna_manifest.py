"""Build an RSNA slice manifest without recursively scanning the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pydicom import dcmread

SUBTYPES = ("intraparenchymal", "intraventricular", "subarachnoid", "subdural", "epidural")
SUBTYPE_COLUMNS = ("IPH", "IVH", "SAH", "SDH", "EDH")
SUBTYPE_TO_COLUMN = dict(zip(SUBTYPES, SUBTYPE_COLUMNS))
HEADER_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber", "ImagePositionPatient", "SOPInstanceUID"]


def base_image_id(value: str) -> str:
    value = str(value).strip()
    for subtype in SUBTYPES:
        suffix = "_" + subtype
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def load_rsna_labels(csv_path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(csv_path, usecols=["ID", "Label"])
    source["image_id"] = source["ID"].map(base_image_id)
    source["subtype"] = source["ID"].str.rsplit("_", n=1).str[-1]
    source["label"] = pd.to_numeric(source["Label"], errors="raise").astype("int8")
    rows = []
    for image_id, group in source.groupby("image_id", sort=False):
        values = {column: 0 for column in SUBTYPE_COLUMNS}
        for _, row in group.iterrows():
            column = SUBTYPE_TO_COLUMN.get(row["subtype"])
            if column is not None:
                values[column] = int(row["label"])
        rows.append({"image_id": image_id, **values})
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise ValueError(f"no RSNA labels found in {csv_path}")
    return labels


def _first_float(value) -> Optional[float]:
    try:
        return float(value[2])
    except (IndexError, TypeError, ValueError):
        return None


def read_dicom_metadata(path: Path) -> Dict[str, object]:
    ds = dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS)
    study = str(getattr(ds, "StudyInstanceUID", ""))
    series = str(getattr(ds, "SeriesInstanceUID", ""))
    instance = getattr(ds, "InstanceNumber", None)
    try:
        instance = int(instance) if instance is not None else None
    except (TypeError, ValueError):
        instance = None
    z_position = _first_float(getattr(ds, "ImagePositionPatient", None))
    order_value = z_position if z_position is not None else instance
    if order_value is None:
        raise ValueError(f"no slice order metadata in {path}")
    return {
        "study_id": study,
        "series_id": series,
        "instance_number": instance,
        "z_position": z_position,
        "order_value": float(order_value),
        "study_key": study or series or path.stem,
    }


def build_manifest(dcm_dir: str | Path, labels_csv: str | Path,
                   output_csv: str | Path, limit: Optional[int] = None,
                   select_json: Optional[str | Path] = None) -> pd.DataFrame:
    """Follow IDs from the CSV; do not enumerate the DICOM directory."""
    dcm_dir = Path(dcm_dir)
    labels = load_rsna_labels(labels_csv)
    if select_json is not None:
        selected = json.loads(Path(select_json).read_text(encoding="utf-8"))
        selected_ids = {base_image_id(key) for key in selected}
        labels = labels[labels["image_id"].isin(selected_ids)].copy()
    if limit is not None:
        labels = labels.head(limit).copy()
    records, missing = [], []
    for row in labels.itertuples(index=False):
        image_id = row.image_id
        path = dcm_dir / f"{image_id}.dcm"
        if not path.exists():
            missing.append(str(path))
            continue
        metadata = read_dicom_metadata(path)
        records.append({
            "image_id": image_id, "dicom_path": str(path.resolve()), **metadata,
            **{column: int(getattr(row, column)) for column in SUBTYPE_COLUMNS},
        })
    if not records:
        detail = f" first missing path: {missing[0]}" if missing else ""
        raise FileNotFoundError(f"no labelled DICOMs found.{detail}")
    manifest = pd.DataFrame(records).sort_values(
        ["study_key", "order_value", "image_id"]
    ).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    if missing:
        print(f"warning: {len(missing)} labelled DICOM paths were missing")
    print(f"wrote {len(manifest)} rows to {output_csv}")
    print(f"studies: {manifest['study_key'].nunique()}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcm-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--select-json")
    args = parser.parse_args()
    build_manifest(**vars(args))


if __name__ == "__main__":
    main()
