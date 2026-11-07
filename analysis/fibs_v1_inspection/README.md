# FIBSv1 inspection (metadata only)

Reproduces the numbers in `notes/fibs-v1-data-inspection.md`.

- Data: `AlignmentResearch/fibs-v1`, 4 parquet files (about 735 MB), downloaded 2026-11-07 from the `main` branch
  (dataset sha `4ad7414f...` per the Hub API at that time; the probe weights were trained on revision `65dccf12`,
  so `main` may differ from the training revision - not checked).
- Environment: throw-away venv with `pyarrow 26.0.1`, `pandas 3.1.1`, `scikit-learn 1.9.1`.
- Run: `python -I analyze.py <outdir>` then `python -I analyze2.py <outdir>`; edit `SRC` at the top to the folder holding the four parquet files.
- Only metadata columns, `completion` length and the parsed message structure are used. No transcript text is stored here.

## Which outputs are valid

- `outputs/summary2.json` and `outputs/family_table.csv` (from `analyze2.py`): valid. Metadata baselines use a real
  `GroupKFold` by scenario (`source`) and by family.
- `outputs/summary.json` (from `analyze.py`): label counts, judge tables and length AUROCs are valid. The first-pass
  "grouped" source-prior AUROCs were removed: `_v32_source_content_md5` is unique per row, so that split was not grouped.
- `outputs/SHA256SUMS` pins the scripts and the four input files.

Nothing here was checked against an independent implementation. Counts were read from a single run.
