"""Probe definition - the ONLY file the research loop edits.

Baseline = the FIBS paper's LinearEnsemble: one logistic-regression probe per layer on the
residual stream at the last token of the final answer, inputs standardized, final score =
mean of the per-layer probabilities.

Interface (called by eval.py):
  fit(train) -> model         train.X[pool]: float32 [n, n_layers, d_model]; train.y: 0/1; train.family: str
  predict(model, data) -> 1-D array of scores, higher = more likely a lie. data.y / data.family are None.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

POOL = "last"
C = 0.01


def fit(train):
    X = train.X[POOL]
    models = []
    for j in range(X.shape[1]):
        scaler = StandardScaler().fit(X[:, j])
        clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X[:, j]), train.y)
        models.append((scaler, clf))
    return models


def predict(models, data):
    X = data.X[POOL]
    probs = [clf.predict_proba(scaler.transform(X[:, j]))[:, 1] for j, (scaler, clf) in enumerate(models)]
    return np.mean(probs, axis=0)
