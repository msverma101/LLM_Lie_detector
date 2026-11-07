"""Probes and baselines. Everything is fit on TRAIN facts only; thresholds come from CALIBRATION facts."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix


class LayerLogisticProbe:
    """One standardised logistic regression per layer; score = mean of per-layer logits."""

    def __init__(self, C: float = 0.05, layer_idx: list[int] | None = None, seed: int = 0):
        self.C, self.layer_idx, self.seed = C, layer_idx, seed
        self.models: list[tuple[np.ndarray, np.ndarray, LogisticRegression]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LayerLogisticProbe":
        idx = self.layer_idx if self.layer_idx is not None else list(range(X.shape[1]))
        self.idx_ = idx
        self.models = []
        for l in idx:
            xl = X[:, l, :].astype(np.float64)
            mu, sd = xl.mean(0), xl.std(0) + 1e-6
            clf = LogisticRegression(C=self.C, max_iter=3000, random_state=self.seed)
            clf.fit((xl - mu) / sd, y)
            self.models.append((mu, sd, clf))
        return self

    def layer_scores(self, X: np.ndarray) -> np.ndarray:
        return np.stack([clf.decision_function((X[:, l, :].astype(np.float64) - mu) / sd)
                         for l, (mu, sd, clf) in zip(self.idx_, self.models)], 1)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.layer_scores(X).mean(1)

    def direction(self, layer_pos: int = 0) -> np.ndarray:
        """Unit direction in RAW activation space for one fitted layer (undo the standardisation)."""
        mu, sd, clf = self.models[layer_pos]
        w = clf.coef_[0] / sd
        return w / np.linalg.norm(w)


def diff_of_means_direction(X_layer: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = X_layer[y == 1].mean(0) - X_layer[y == 0].mean(0)
    return d / np.linalg.norm(d)


class TextBaseline:
    """Equal-information text baseline: TF-IDF (word + char n-grams) of system+user+assistant, plus length."""

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)

    @staticmethod
    def _len(texts):
        return csr_matrix(np.log1p(np.array([[len(t)] for t in texts], float)))

    def fit(self, texts, y):
        X = hstack([self.w.fit_transform(texts), self.c.fit_transform(texts), self._len(texts)]).tocsr()
        self.clf.fit(X, y)
        return self

    def score(self, texts):
        X = hstack([self.w.transform(texts), self.c.transform(texts), self._len(texts)]).tocsr()
        return self.clf.decision_function(X)
