# Research findings

Literature research for the LLM lie-detection hackathon project (7 Nov 2026): what is state of the art, whether
anything is at par with or better than **FIBS** (activation probes) and **Weight-Space Programming** (LLMs reading and
editing weights), and what we could build instead.

## Start here

| File | What it is |
|---|---|
| [`findings-2026-11-07.md`](findings-2026-11-07.md) | **Summary.** Verdicts per paper with full-text checks and code links, and what to build |
| [`fusion-architecture.md`](fusion-architecture.md) | Design proposal: one monitor fusing FIBS probe, honesty adapter and a weight channel, with a proof that fusion is never worse than the best single method and a one-hour test plan |
| [`sota-vs-fibs-and-wsp.md`](sota-vs-fibs-and-wsp.md) | Detailed comparison tables and five project ideas ("find the liar heads" recommended) |
| [`state-of-the-art.md`](state-of-the-art.md) | About 200 papers and lab posts grouped by topic, with how the search was done |

## Key results in one paragraph

No paper beats FIBS on its own agent-sabotage benchmarks. The strongest challenger is an ICLR 2027 submission on
**honesty adapters**: on lying models built with an unseen recipe (AuditBench) it reaches 0.90 AUROC against 0.66 for a
FIBS-architecture probe, while only matching probes on ordinary lies. On the weight side, WSP is matched or beaten on
rigour (Weight Oracles), on real model scale (weight-diff readers, parameter decomposition) and by plain fine-tuning in
a fair replication; the newest critique shows the gains come from letting the LLM *compute* with weights, not read them.

## Tools

`tools/find_sota.py` searches arXiv, OpenAlex and OpenReview (no API keys needed), skips papers already listed here or
in `literature/`, and flags new papers that report scores near or above FIBS.

```bash
python3 research-findings/tools/find_sota.py --aspect detection --days 60
python3 research-findings/tools/find_sota.py --aspect weights --days 365 --out research-findings/script-output/weights.md
```

`script-output/` holds the runs from 7 Nov (two aspects × last 60 and 365 days). They are keyword-filtered candidate
lists: read them, don't trust them blindly.

## Caveats

Most numbers come from paper abstracts; full texts were checked for FIBS, the honesty-adapter paper, Weight Oracles and
about 15 others (marked in the files). Benchmarks differ between papers, so "better" is always on a stated axis.
