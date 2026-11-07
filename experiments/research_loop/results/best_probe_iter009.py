"""Best probe from the research loop (iteration 9), written by Claude Opus 5.5.
Copied verbatim from runs/iter_009.py of the Colab run on 2026-11-07.

LinearEnsemble over pools and layers: one logistic-regression probe per (pool, layer) on
standardized activations. Each probe is weighted by its cross-family generalization,
estimated with grouped (by family) cross-validation on the training set. The final score
is the weighted mean of the per-probe probabilities.

Interface (called by eval.py):
  fit(train) -> model         train.X[pool]: float32 [n, n_layers, d_model]; train.y: 0/1; train.family: str
  predict(model, data) -> 1-D array of scores, higher = more likely a lie. data.y / data.family are None.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

POOLS = ["last", "mean_answer", "mean_turn"]
C = 0.01
N_FOLDS = 7
TEMP = 0.01


def _fit_one(X, y):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=0).fit(scaler.transform(X), y)
    return scaler, clf


def _macro_auc(scores, y, fam):
    aucs = []
    for f in np.unique(fam):
        m = fam == f
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], scores[m]))
    return float(np.mean(aucs)) if aucs else 0.5


def fit(train):
    y = np.asarray(train.y)
    fam = np.asarray(train.family)
    n_fam = len(np.unique(fam))
    gkf = GroupKFold(n_splits=min(N_FOLDS, n_fam))
    splits = list(gkf.split(np.zeros(len(y)), y, fam))

    models, cv_aucs = [], []
    for pool in POOLS:
        X = train.X[pool]
        for j in range(X.shape[1]):
            Xj = X[:, j].astype(np.float64)
            oof = np.zeros(len(y))
            for tr, te in splits:
                sc, clf = _fit_one(Xj[tr], y[tr])
                oof[te] = clf.decision_function(sc.transform(Xj[te]))
            cv_aucs.append(_macro_auc(oof, y, fam))
            scaler, clf = _fit_one(Xj, y)
            models.append((pool, j, scaler, clf))

    cv_aucs = np.array(cv_aucs)
    w = np.exp((cv_aucs - cv_aucs.max()) / TEMP)
    w = w / w.sum()
    return {"models": models, "weights": w, "cv_aucs": cv_aucs}


def predict(model, data):
    probs = []
    for pool, j, scaler, clf in model["models"]:
        Xj = data.X[pool][:, j].astype(np.float64)
        probs.append(clf.predict_proba(scaler.transform(Xj))[:, 1])
    probs = np.stack(probs, axis=0)
    return np.tensordot(model["weights"], probs, axes=1)
