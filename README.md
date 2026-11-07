# LLM_Lie_detector

## Plumbline: lies in a model, honestly measured (video)

<video src="presentation/plumbline-paper-video/plumbline-lie-detection-paper.en.mp4" poster="presentation/plumbline-paper-video/poster.jpg" controls width="720"></video>

[![Plumbline: lies in a model, honestly measured. Open the English video](presentation/plumbline-paper-video/poster.jpg)](presentation/plumbline-paper-video/plumbline-lie-detection-paper.en.mp4)

English, 9:47, built like a paper (abstract, introduction, methodology, results, conclusion). It explains the Plumbline procedure: a FIBS probe on the activations of a frozen Qwen3.5-2B, LLM-proposed edits to the probe's weights (WSP), and a fixed measurement discipline (preregistered plan, one-shot test, certified false-alarm threshold), and reports the measured result: the LLM-proposed edits showed no advantage over random edits. German version: [plumbline-lie-detection-paper.de.mp4](presentation/plumbline-paper-video/plumbline-lie-detection-paper.de.mp4). Details and sources: [presentation/plumbline-paper-video/README.md](presentation/plumbline-paper-video/README.md).

## Video explainer

[![FIBS × WSP: Better detectors. Frozen LLMs.](presentation/fibs-wsp-explainer/poster.jpg)](presentation/fibs-wsp-explainer/README.md)

[Watch or download the two-minute English video](presentation/fibs-wsp-explainer/README.md), with subtitles, transcript, and research references. It explains the proposed detector-only methodology and ends with findings from the original WSP report.

## Ten documented detector experiments

[Results, hypotheses, controls and reproducibility bundle](experiments/diagnostic_ten/REPORT.md). Ten real candidates were evaluated with a frozen LLM. The WSP improvement hypothesis was not supported; ordinary fitting showed a modest favourable point estimate. [Five-criterion evidence](experiments/diagnostic_ten/CRITERIA.md).
