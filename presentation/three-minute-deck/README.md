# 3-minute closing talk: "Can We Catch a Model Lying?"

Animated deck (10 slides, ~3:00 with speaker notes):
https://claude.ai/artifact/7x72Y6X3ieuBGVJUNrj7Fa (private until shared from its Share menu)

Source: `project/deck.json` (order, sections, fonts) and one file per slide in `project/slides/`.
Each slide's `<aside>` holds the speaker notes with timings.

| # | Slide | Time |
|---|---|---|
| 1 | Cover with an animated lie-meter | 0:00 |
| 2 | The bet: FIBS = detector, WSP = detective | 0:12 |
| 3 | The afternoon as git log | 0:30 |
| 4 | Experiment 1: Claude rewrites the probe, 0.806 → 0.845 (animated chart) | 0:50 |
| 5 | What Claude found | 1:15 |
| 6 | Experiment 2: Claude edits weights, negative result | 1:35 |
| 7 | The interface: a lie detector that debugs itself (animated loop) | 1:55 |
| 8 | Case #1 (WSP loop session 1): caving AUROC 0.58 → 0.78, selection only | 2:15 |
| 9 | Literature sweep: ~200 papers | 2:30 |
| 10 | Four lessons + thanks | 2:45 |

The interface slide (7) is a proposal, not a measured result. Experiment 1 supports the
"change the formula" half; Experiment 2 shows that weight edits without case files and computation
were no better than random edits.
