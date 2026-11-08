# LLM_Lie_detector

[![Plumbline explainer, animated preview (30 s). Click to open the English video](presentation/plumbline-paper-video/plumbline-preview.en.gif)](presentation/plumbline-paper-video/plumbline-lie-detection-paper.en.mp4)

## Plumbline: lies in a model, honestly measured (video)

English, 9:47, built like a paper (abstract, introduction, methodology, results, conclusion). It explains the Plumbline procedure: a FIBS probe on the activations of a frozen Qwen3.5-2B, LLM-proposed edits to the probe's weights (WSP), and a fixed measurement discipline (preregistered plan, one-shot test, certified false-alarm threshold), and reports the measured result: the LLM-proposed edits showed no advantage over random edits. German version: [plumbline-lie-detection-paper.de.mp4](presentation/plumbline-paper-video/plumbline-lie-detection-paper.de.mp4). Details and sources: [presentation/plumbline-paper-video/README.md](presentation/plumbline-paper-video/README.md).

## Video explainer

[![FIBS × WSP: Better detectors. Frozen LLMs.](presentation/fibs-wsp-explainer/poster.jpg)](presentation/fibs-wsp-explainer/README.md)

[Watch or download the two-minute English video](presentation/fibs-wsp-explainer/README.md), with subtitles, transcript, and research references. It explains the proposed detector-only methodology and ends with findings from the original WSP report.

## Ten documented detector experiments

[Results, hypotheses, controls and reproducibility bundle](experiments/diagnostic_ten/REPORT.md). Ten real candidates were evaluated with a frozen LLM. The WSP improvement hypothesis was not supported; ordinary fitting showed a modest favourable point estimate. [Five-criterion evidence](experiments/diagnostic_ten/CRITERIA.md).

## Harness and preregistered experiment

[Documentation of the `liedet` package](src/liedet/README.md): published FIBS probes, calibrated evaluation, LLM-proposed probe edits (WSP) and the first preregistered run (WSP not supported on 65 held-out facts). Plan, result note and raw data are under [`docs/`](docs/) and `docs/data/experiments/fibs-wsp-2026-11-07/`.


### Adaptive WSP follow-up (8 November 2026)

A fresh 3,400-example study completed ten adaptive detector-only coefficient edits with fixed features, separate calibration and a frozen final test. The selected WSP detector did not establish an improvement: recall 41.00% versus 42.33% original, with actual FPR 2.00% versus 2.67%. See the [full report](experiments/adaptive_wsp/REPORT.md), [five-criterion evidence audit](experiments/adaptive_wsp/CRITERIA.md) and [complete raw-data replay archive](https://github.com/msverma101/LLM_Lie_detector/releases/tag/adaptive-wsp-2026-11-08).

### Source-grounded WSP skill and fresh validation (8 November 2026)

The [detector-weight-space-debugging skill](.agents/skills/detector-weight-space-debugging/SKILL.md) incorporates a detailed audit of the original authors' procedure, direct detector-only edits, falsifiable hypotheses, controls and independent confirmation. In a fresh 3,191-row study, a natural intervention failed its diagnostic preservation check and was rejected; the unchanged detector remained selected. A separately registered engineered coefficient fault was localized and repaired, increasing detections from 194 to 232 of 564 positives. This demonstrates injected-fault recovery, not improvement over the uncorrupted detector. See the [completed report](experiments/wsp_skill_validation/REPORT.md) and [complete replay archive](https://github.com/msverma101/LLM_Lie_detector/releases/tag/wsp-skill-validation-2026-11-08).
