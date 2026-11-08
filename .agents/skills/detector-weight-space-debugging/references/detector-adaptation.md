# Separate-detector adaptation

## Computation and repair limits

For cached layer activation x_l, q_l=(x_l-mu_l)/sd_l and z_l=w_l.T*q_l+b_l.
The existing detector score is mean_l sigmoid(z_l). Only w_l/b_l may change.
The monitored LLM stays frozen. A linear last-token probe can recombine available
signals; it cannot create absent information or establish the model's deceptive
intent. FIBS completion labels are not subjective beliefs of the monitored model.

A coefficient contribution explains the arithmetic, not semantic meaning or a
causal mechanism in the LLM. Direct detector interventions demonstrate effects on
its scores; generalization still needs independent cases and controls.

## Possible directional repair, not a default algorithm

If matched evidence establishes a nuisance direction u in standardized input
space, with ||u||=1, an explicit detector edit could be:

    w' = w - alpha * u * (u.T @ w)
    z'(x)-z(x) = -alpha * (w.T@u) * (u.T@q(x)).

Alpha1 removes the parallel coefficient component; larger values reverse it.
This adapts the role/transposition of a direction: it is not the same operation as
editing Qwen's MLP. If u also carries truth signal, deleting it can hurt detection.

Matched cases should vary the proposed nuisance while holding label/content as
stable as possible. Arbitrary false-positive/true-negative contrasts change many
factors. A mean honest-versus-deceptive contrast estimates a classifier; fitting
new weights from it automatically is a statistical method, not an explanatory WSP
result. Disclose any data-assisted contrast and freeze it before transfer tests.
Allow other justified direct edits; do not constrain every hypothesis to this formula.

## Corrections to the preceding experiment

Our six last-token linear probes were a disclosed alternative to the published
FIBS EFC checkpoint. Its selected attenuation reduced a whole probe to25% strength.
Although the general WSP definition is broad, original Qwen Experiment8 explicitly
rejected pure whole-tensor scaling. Describing that attenuation as faithful to
the specific repair protocol was too broad.

Local contribution predictions did not establish a common semantic failure.
Several were arithmetic predictions on the examples used to design the patch.
Attenuation lost test detections while reducing false alarms. Since thresholds
were independently calibrated, this does not establish matched-FPR inferiority.
The selection rounding bug was fixed before test opening and is not a demonstrated
cause of that result. Preserve old records; do not promote a favourable old ablation
as a fresh confirmed improvement.

## Five evidence categories for this adaptation

1. Software: immutable representation, valid coefficient edits and score replay.
2. Mechanism: diagnostic discovery/transfer, predicted intervention effects,
   dose/isolation/null controls and bounded harm to correct detections.
3. Selection: constrained primary metric with uncertainty, not an AUROC tradeoff.
4. Efficacy: candidate frozen before new calibration/test, paired group intervals,
   actual false alarms, no post-test tuning.
5. Generality: independent agent studies/models/domains; distinguish these from
   new rows from familiar families or replaying the same saved study.

These are explicit evidence categories, not a validated quality score or a claim
to reproduce every earlier conversational wording of "five criteria". Passing
one cannot substitute for another.

With n honest calibration examples, t is order statistic
ceil((n+1)*(1-alpha)); predict positive only for s>t. If the rank exceeds n,
t=infinity and no detection is certified: report insufficient calibration rather
than silently changing alpha. Each arm has its own t and actual FPR.

Resample source groups jointly for both arms and resample calibration as well as
test. State whether family stratification conditions on familiar sources. Row-wise
binomial bounds alone do not address clusters, calibration uncertainty or shift.

## Skill validation versus detector efficacy

An engineered-fault benchmark may corrupt detector weights and test diagnosis/
recovery on separate real activation rows. Include the uncorrupted checkpoint and
a wrong intervention. Hide the corruption definition from an independent evaluator
when testing discovery. Recovery proves the procedure repaired that injected fault;
it does not establish a similar natural fault or superiority to the uncorrupted
baseline. Synthetic fixtures test only mathematical/software behavior.

Abstention after an unsupported mechanism is appropriate. Report skill/software
validation, engineered repair and natural-data efficacy as separate findings.
