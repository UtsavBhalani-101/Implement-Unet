"""Dataset-free verification of manifest and three-slice preprocessing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom import dcmwrite
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from PIL import Image

from .preprocess_rsna import preprocess_study
from .rsna_manifest import build_manifest


def write_dicom(path: Path, value: int, study_uid: str, instance: int) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, generate_uid()
    ds.InstanceNumber, ds.Rows, ds.Columns = instance, 8, 8
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.full((8, 8), value, dtype=np.uint16).tobytes()
    dcmwrite(str(path), ds, write_like_original=False)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root, dcm_dir = Path(temp), Path(temp) / "stage_2_train"
        dcm_dir.mkdir()
        rows, study_uid = [], generate_uid()
        subtypes = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
        for index in range(3):
            image_id = f"ID_test{index}"
            write_dicom(dcm_dir / f"{image_id}.dcm", index * 100, study_uid, index + 1)
            rows.extend({"ID": f"{image_id}_{subtype}", "Label": int(subtype == "intraparenchymal")} for subtype in subtypes)
        labels = root / "stage_2_train.csv"
        pd.DataFrame(rows).to_csv(labels, index=False)
        manifest = build_manifest(dcm_dir, labels, root / "manifest.csv")
        assert len(manifest) == 3 and manifest["study_key"].nunique() == 1
        assert manifest["order_value"].tolist() == [1.0, 2.0, 3.0]
        outputs = preprocess_study(manifest, root / "png", resize=16)
        image = np.asarray(Image.open(outputs.iloc[1]["png_path"]))
        assert len(outputs) == 3 and outputs["IPH"].tolist() == [1, 1, 1]
        assert image.shape == (16, 16, 3) and image.dtype == np.uint8
        print("PASS")
        print("manifest rows:", len(manifest))
        print("study keys:", manifest["study_key"].nunique())
        print("PNG shape:", image.shape)


if __name__ == "__main__":
    main()
