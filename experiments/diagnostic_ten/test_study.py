"""Software-only integration checks. Generated fixtures are never research results."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("study", Path(__file__).with_name("study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class StudyChecks(unittest.TestCase):
    def test_finite_sample_threshold(self):
        self.assertEqual(study.threshold(np.arange(100), 0.05), 95)
        self.assertTrue(np.isinf(study.threshold(np.arange(8), 0.01)))
        self.assertFalse(np.any(np.array([0.5, 0.5]) > study.threshold([0.5] * 100, 0.05)))

    def test_full_fixture_pipeline_and_guards(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            capture, run = root / "capture", root / "run"
            capture.mkdir()
            rng = np.random.default_rng(9)
            frames, arrays = [], []
            for part, n in [("fit", 100), ("inspect", 80), ("selection", 80), ("calib", 120), ("test", 120)]:
                y = np.zeros(n, dtype=int) if part == "calib" else np.arange(n) % 2
                X = rng.normal(size=(n, 3, 6, 12))
                X[:, :, :, 0] += (y * 2 - 1)[:, None, None] * 0.6
                meta = pd.DataFrame({"row_id": [part + str(i) for i in range(n)], "group_id": [part + str(i // 2) for i in range(n)],
                                     "partition": part, "family": ["f" + str(i % 3) for i in range(n)], "label": y,
                                     "quadrant": ["honest_deceptive" if v else "deceptive_honest" for v in y], "answer_tokens": rng.integers(3, 30, n)})
                frames.append(meta)
                arrays.append(X)
                cases = meta.copy()
                cases["messages_json"] = '[{"role":"user","content":"Synthetic software fixture."}]'
                cases["completion"] = "Synthetic fixture, not an empirical model response."
                cases.to_json(capture / f"cases_{part}.jsonl", orient="records", lines=True)
            pd.concat(frames).to_csv(capture / "meta.csv", index=False)
            X = np.concatenate(arrays)
            for p, name in enumerate(study.POOLS):
                np.save(capture / f"acts_{name}.npy", X[:, p])
            (capture / "capture_manifest.json").write_text('{"software_fixture": true}')
            args = SimpleNamespace(capture=str(capture), run=str(run), spec=None)
            study.prepare(args)
            with self.assertRaises(AssertionError):
                study.prepare(args)
            for i in range(1, 11):
                item = {"attempt": i, "hypothesis": "Fixture-only integration", "falsifying_prediction": "Fixture-only",
                        "rationale": "Exercise artifact and leakage guards", "category": "software_fixture",
                        "branches": [{"pool": study.POOLS[(i - 1) % 3], "layers": [1, 2, 3]}]}
                path = root / f"spec_{i}.json"
                path.write_text(json.dumps(item))
                args.spec = str(path)
                study.attempt(args)
            with self.assertRaises(AssertionError):
                study.attempt(args)
            study.freeze(args)
            study.final(args)
            study.replay(args)
            with self.assertRaises(AssertionError):
                study.final(args)
            self.assertEqual(len(list(run.glob("attempt_*/spec.json"))), 10)
            self.assertEqual(json.loads((run / "manifest.json").read_text())["completed_attempts"], 10)


if __name__ == "__main__":
    unittest.main()
