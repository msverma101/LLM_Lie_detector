# Experiment data

Raw measurement data, one directory per experiment: `data/experiments/<slug>/`.
Each directory holds the raw files plus `MANIFEST.sha256` (`sha256sum * > MANIFEST.sha256`).
The experiment itself is recorded through the gates (see CONCEPT-0001):
a preregistration `plan`, then a result `note` that links here with `--link evidence=...`.
Never create plan/note files by hand; use `kb.py new`.
