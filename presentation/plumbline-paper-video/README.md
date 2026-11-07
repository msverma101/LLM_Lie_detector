# Plumbline: lies in a model, honestly measured

Paper-style explainer video (abstract, introduction, methodology, results, conclusion) of the Plumbline procedure:
a FIBS probe reads the activations of a frozen Qwen3.5-2B, a second language model proposes edits to the probe's weights
(Weight-Space Programming, WSP), and a fixed measurement discipline decides whether an edit really helps.
"Plumbline" is a working name chosen for this video, not an established term.

| File | Language | Duration | Size | Format |
|---|---|---|---|---|
| `plumbline-lie-detection-paper.en.mp4` | English | 586.7 s (9:47) | 30.4 MB | 1920x1080, 30 fps, H.264, AAC, -16 LUFS |
| `plumbline-lie-detection-paper.de.mp4` | German | 639.5 s (10:39) | 32.7 MB | 1920x1080, 30 fps, H.264, AAC, -16 LUFS |
| `poster.jpg` | | | 57 KB | frame of the title slide with a play button |

## Where the numbers come from

Every number in the video is read at build time from the raw files of the run of 2026-11-07, published on the branch
`feature/liedet-fibs-wsp-harness` under `docs/data/experiments/fibs-wsp-2026-11-07/` (`final_test.json`,
`secondary_test_analysis.json`, `dev_report_followup.json`, `fibs_parity.json`, `arms_dev_summary.json`, `meta.json`).
They are NOT the numbers of `experiments/diagnostic_ten/` on `main`, which is a separate set of experiments.

Main result shown: on 65 held-out facts, LLM-proposed edits minus the best ordinary method (random edits) =
-1.5 percentage points detection rate, 95% interval -8.5 to +6.2, so no advantage was shown; both edit arms raised the
detection rate (about 11% to 21-22%) and the false-alarm rate (0.8% to 3-4%). Limits stated in the video: one small
model, one run, labels by construction, wide intervals; the procedure measures, it does not prevent lying and does not
cover hallucinations.

## Production

Made with the CorvinTV slide pipeline (web slides, light theme with a white background), narration by OpenAI tts-1-hd
(voice onyx), loudness normalised to -16 LUFS. The audio was measured, not listened to by a person. FIBS and WSP are
described only as their sources define them (FIBS paper, arXiv 2610.12445; WSP blog post of 2 November 2026,
author-reported, not peer reviewed).
