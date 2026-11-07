# Five-criterion evidence index

These are evidence-based workshop judgments, not official jury scores. A negative scientific outcome can satisfy the reporting criteria.

| Criterion | Completed evidence | Remaining limit |
|---|---|---|
| Problem | [Protocol](PROTOCOL.md): frozen-LLM, detector-only question, H1/H2 and false-alarm target fixed | FIBS supplied-answer labels do not establish all deceptive intent |
| Loop — partial | [Diagnosis](DIAGNOSIS.md), ten preregistered hypotheses, saved candidates, feedback and keep/revert records in the archive | Fixed batch; every candidate starts from theta0. Feedback did not drive subsequent hypotheses or incumbent edits. Error causes were not established; see [audit](POSTHOC_AUDIT.md) |
| Evidence | [Report](REPORT.md), once-opened group holdout, matched ordinary controls, counts and joint calibration/test bootstrap intervals | WSP benefit not demonstrated; one draw, familiar target families and sparse cells |
| Honesty | Negative H1/H2 reported; actual FPR violations, historical exposure and identical random control disclosed | No universal efficacy or deployment-safety claim |
| Reproducibility | All checkpoints and score arrays archived; score replay passes; source and model/data revisions pinned | Raw activations need regeneration; independent end-to-end reproduction remains unverified |

All five criteria have pilot artifacts, but that is not full satisfaction of the intended methodology. The adaptive loop is only partially implemented, evidence is one negative pilot, and independent end-to-end reproducibility is unverified. The WSP improvement hypothesis did not succeed in this run.
