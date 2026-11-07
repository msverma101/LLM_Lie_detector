# FIBS × WSP — Better detectors. Frozen LLMs.

English narration · 2:00 · proposed methodology followed by results from the original WSP report.

## 00:00–00:14 · Understand errors. Improve the detector.

Our approach connects FIBS with weight-space programming. A separate detector reads a frozen language model’s internal activations. We aim to catch more lies with fewer false alarms. Only the detector is edited; the language model stays unchanged.

## 00:14–00:32 · Every error needs context.

First, inspect false alarms and missed lies. Each case needs the full conversation, the final answer, and a checkable reason for its label. Correctly classified cases provide controls. This separates detector errors from ambiguous labels or missing context.

## 00:32–00:50 · Turn a pattern into a test.

One hypothesis: the detector mistakes an earlier lie for a currently honest correction. Test matched cases with truthful or deceptive history, and truthful or deceptive final answers. State the prediction before editing. This is a proposed diagnostic test, not an established finding.

## 00:50–01:10 · Edit the detector. Keep the LLM frozen.

After testing the explanation, Claude examines layer contributions and writes a targeted change to the detector’s weights. The monitored language model stays frozen. Compare the original detector, the direct edit, ordinary optimization, and random edits. Check the targeted errors and possible side effects.

## 01:10–01:28 · A loop with clear boundaries.

The loop is: inspect errors, test an explanation, edit, and measure again. Keep development, selection, and calibration separate. The final evaluation needs genuinely untouched cases. Report detection and actual false alarms. Save weights, predictions, and decisions so others can check the work.

## 01:28–02:00 · What the original WSP report shows.

In the original WSP report, Qwen’s MATH five hundred score rises from seven point nine to forty-eight point one percent. The edit changes just zero point zero one six percent of the language model’s weights, reducing repetition loops. Gemma shows no clear gain. This demonstrates feasibility, not better lie detection. Our proposal transfers the editing approach to the separate detector. Whether that improves detection is the hypothesis this loop must test.
