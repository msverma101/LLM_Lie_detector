"""Behavioral/software invariants; synthetic fixtures are not efficacy findings."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".agents/skills/detector-weight-space-debugging/scripts"


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


audit, evaluation = module("audit_edit"), module("paired_eval")


class SkillChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.parent = {"W": rng.normal(size=(3, 20)), "b": np.ones(3),
                       "mu": np.zeros((3, 20)), "sd": np.ones((3, 20))}

    def copy(self):
        return {k: v.copy() for k, v in self.parent.items()}

    def test_reject_global_affine_confidence_patch_and_accept_control_label(self):
        t = self.copy()
        t["W"][0] *= .25
        t["b"][0] = .123
        self.assertFalse(audit.inspect(self.parent, t)["allowed"])
        self.assertTrue(audit.inspect(self.parent, t, "calibration-control")["allowed"])

    def test_local_edit_cannot_hide_representation_change(self):
        t = self.copy()
        t["W"][0, 3] = 0
        self.assertTrue(audit.inspect(self.parent, t)["allowed"])
        t["sd"][0, 0] = 1.01
        self.assertFalse(audit.inspect(self.parent, t)["allowed"])

    def test_finite_sample_cutoff_and_strict_ties(self):
        self.assertEqual(evaluation.cutoff(np.arange(39), .025), 38)
        self.assertTrue(np.isinf(evaluation.cutoff(np.arange(38), .025)))
        self.assertEqual(evaluation.counts([0, 1], [38, 39], 38)["fp"], 0)
        self.assertEqual(evaluation.counts([0, 1], [38, 39], 38)["tp"], 1)

    def test_pairing_uses_ids_and_rejects_label_changes(self):
        a = pd.DataFrame({"row_id": ["a", "b"], "group_id": ["a", "b"], "family": ["f", "f"],
                          "label": [0, 1], "score": [.1, .9]})
        paired = evaluation.align(a, a.iloc[::-1])
        np.testing.assert_array_equal(paired.original.to_numpy(), paired.candidate.to_numpy())
        b = a.copy()
        b.loc[0, "label"] = 1
        with self.assertRaises(ValueError):
            evaluation.align(a, b)

    def test_reject_calibration_overlap(self):
        c = pd.DataFrame({"row_id": [str(i) for i in range(100)], "group_id": [str(i) for i in range(100)],
                          "family": "f", "label": 0, "original": np.linspace(0, 1, 100),
                          "candidate": np.linspace(0, 1, 100)})
        e = c.iloc[:10].copy()
        e.loc[0, "label"] = 1
        with self.assertRaises(ValueError):
            evaluation.compare(c, e, draws=100)

    def test_identical_detector_never_claims_improvement(self):
        rng = np.random.default_rng(80)
        c = pd.DataFrame({"row_id": ["c" + str(i) for i in range(100)],
                          "group_id": ["c" + str(i // 2) for i in range(100)], "family": "f", "label": 0,
                          "original": rng.uniform(size=100)})
        c["candidate"] = c.original
        e = pd.DataFrame({"row_id": ["e" + str(i) for i in range(100)],
                          "group_id": ["e" + str(i // 2) for i in range(100)], "family": "f",
                          "label": np.tile([0, 1], 50), "original": rng.uniform(size=100)})
        e["candidate"] = e.original
        result = evaluation.compare(c, e, draws=100)
        self.assertFalse(result["confirmed_improvement"])
        self.assertEqual(result["joint_group_bootstrap"]["recall_difference_95"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
