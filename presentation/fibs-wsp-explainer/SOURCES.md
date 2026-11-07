# FIBS × WSP — Better detectors. Frozen LLMs.

Two-minute English animated explainer. The detector-only methodology is a proposal. The final section contains only results from the original WSP report. It contains no results from our repository experiments.

## Claims in the ending

- Qwen3.5-0.8B, MATH-500: 7.9% before editing, 48.1% after the 50-feedback-call condition.
- The original intervention changed 131,072 parameters, approximately 0.016% of the LLM's weights, and reduced repetition loops.
- The Gemma 3 270M control did not establish a clear improvement.
- These results demonstrate feasibility in those tasks. They do not establish that WSP improves lie detection.
- Our proposed intervention is different: edit a separate activation detector while keeping the monitored LLM frozen.

Source: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, November 2, 2026, Experiment 8 and its Gemma control. [Original report](https://rikiyat.github.io/blog/weight-space-programming/). This is a research blog report, not described here as a peer-reviewed paper. The findings were source-checked; they were not independently reproduced for this video.

## Methodology background

Oskar J. Hollinsworth, Alex F. Spies, Tigist Diriba, Adam Gleave and Chris Cundy, *Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception*, arXiv:2610.12445v4. [Paper](https://arxiv.org/pdf/2610.12445v4).

The example about a past lie and a current honest correction is illustrative. It is a proposed diagnostic hypothesis, not a reported finding. The detector-weight numbers shown in the animation are schematic.

## Production

- Original vector animation rendered with Manim; synthetic English narration; original synthesized ambient bed.
- Skills used: scripting-and-storyboarding for the scene plan, manim-video for the animated explainer, and ffmpeg-skill for audio assembly, normalization, inspection and delivery checks.
- No cloned voice, presenter likeness, borrowed music, or stock footage.
- Separate English subtitles accompany the MP4 and are also embedded as an optional subtitle track.
- Research status and intervention scope were updated to reflect the user's corrections: English narration, no own experiment results, and no intervention in the monitored LLM in the proposed study.
