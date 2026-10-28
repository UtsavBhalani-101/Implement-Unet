# RSNA metadata and preprocessing

The standard RSNA CSV is used as a label lookup table. The manifest builder
derives each DICOM path directly from the image ID and reads only DICOM headers
to obtain study/series/slice order. It does not recursively enumerate the
700,000-file training directory.

From `author_repo`:

```powershell
python -m data_pipeline.verify_stage_3_4
```

For the real dataset, begin with a small limit:

```powershell
python -m data_pipeline.rsna_manifest `
  --dcm-dir D:\path\to\stage_2_train `
  --labels-csv D:\path\to\stage_2_train.csv `
  --output-csv D:\path\to\rsna_manifest.csv `
  --limit 100

python -m data_pipeline.preprocess_rsna `
  --manifest-csv D:\path\to\rsna_manifest.csv `
  --output-dir D:\path\to\png `
  --output-manifest D:\path\to\png_manifest.csv `
  --limit-studies 1
```
