"""Command line: build-data -> capture -> evaluate -> weights (run-all chains them)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from . import __version__
from . import data as D
from . import experiment as ex


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out: Path) -> None:
    lines = [f"{sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "MANIFEST.sha256"]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def load_rows(data_dir: Path) -> list[D.Row]:
    return [D.Row(**json.loads(l)) for l in (data_dir / "rows.jsonl").read_text().splitlines()]


def load_facts(data_dir: Path) -> list[D.Fact]:
    return [D.Fact(**f) for f in json.loads((data_dir / "facts.json").read_text())]


def cmd_build_data(a) -> int:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    facts = D.build_facts(a.seed, a.n_arith)
    if a.limit_facts:   # deterministic family-preserving subsample for smoke runs
        import random
        keep = set(random.Random(a.seed).sample([f.fact_id for f in facts], min(a.limit_facts, len(facts))))
        facts = [f for f in facts if f.fact_id in keep]
    rows = D.build_rows(facts)
    tr, ca, te = D.split_by_fact(facts, a.seed)
    (out / "facts.json").write_text(json.dumps([f.__dict__ for f in facts], indent=1))
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r.to_dict()) for r in rows) + "\n")
    (out / "split.json").write_text(json.dumps({"train": tr, "calib": ca, "test": te, "seed": a.seed}, indent=1))
    write_manifest(out)
    print(f"facts={len(facts)} rows={len(rows)} train/calib/test facts={len(tr)}/{len(ca)}/{len(te)} -> {out}")
    return 0


def cmd_capture(a) -> int:
    from .capture import Capturer
    data_dir, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    facts, rows = load_facts(data_dir), load_rows(data_dir)
    cap = Capturer(a.model, dtype=a.dtype)
    margins = {}
    for i, f in enumerate(facts):
        margins[f.fact_id] = cap.belief_margin(f.question, f.true_answer, f.false_answer)
        if i % 20 == 0:
            print(f"belief {i}/{len(facts)}", file=sys.stderr, flush=True)
    kept = {f for f, m in margins.items() if m > 0}
    rows_kept = [r for r in rows if r.fact_id in kept] if a.skip_unbelieved else rows
    feats = cap.capture_rows(rows_kept, keep_tokens=a.tokens, progress=lambda n, t: print(f"capture {n}/{t}", file=sys.stderr, flush=True)
                             if n % 25 == 0 else None)
    np.savez_compressed(out / "features.npz", **feats)
    (out / "rows_used.jsonl").write_text("\n".join(json.dumps(r.to_dict()) for r in rows_kept) + "\n")
    (out / "belief_margins.json").write_text(json.dumps(margins, indent=1))
    (out / "meta.json").write_text(json.dumps({
        "liedet": __version__, "model": a.model, "dtype": a.dtype, "layers": list(cap.layers),
        "d_model": cap.d_model, "n_layers": cap.n_layers, "python": platform.python_version(),
        "torch": __import__("torch").__version__, "transformers": __import__("transformers").__version__,
        "n_rows": len(rows_kept), "n_facts": len(facts), "rows_sha256": sha256(data_dir / "rows.jsonl"),
        "belief_kept": len(kept),
    }, indent=1))
    write_manifest(out)
    print(f"captured rows={len(rows_kept)} belief_kept={len(kept)}/{len(facts)} -> {out}")
    return 0


def _load_features(feat_dir: Path):
    z = np.load(feat_dir / "features.npz", allow_pickle=False)
    rows = [D.Row(**json.loads(l)) for l in (feat_dir / "rows_used.jsonl").read_text().splitlines()]
    assert list(z["row_ids"]) == [r.row_id for r in rows], "feature rows do not match rows_used.jsonl"
    feats = {k: z[k] for k in ("mean", "last", "lp_neutral")}
    feats.update({k: z[k] for k in ("tok_acts", "tok_offsets") if k in z.files})
    margins = json.loads((feat_dir / "belief_margins.json").read_text())
    return rows, feats, margins


def cmd_evaluate(a) -> int:
    data_dir, feat_dir, out = Path(a.data), Path(a.features), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, feats, margins = _load_features(feat_dir)
    facts = load_facts(data_dir)
    split = json.loads((data_dir / "split.json").read_text())
    cfg = ex.Config(alpha=a.alpha, pooling=a.pooling, n_boot=a.n_boot, seed=a.seed)
    results = {}
    for variant in ("answer", "followup"):
        results[f"in_distribution/{variant}"] = ex.run(rows, feats, margins, split, cfg, variant)
    fams = sorted({f.family for f in facts})
    for fam in fams:
        sp = ex.cross_family_split(facts, fam, a.seed)
        results[f"held_out_family={fam}/answer"] = ex.run(rows, feats, margins, sp, cfg, "answer")
    (out / "results.json").write_text(ex.to_json(results))
    (out / "report.md").write_text(ex.report_md(results))
    write_manifest(out)
    for k, v in results.items():
        p = v["primary"]
        print(f"{k}: probe_auroc={v['methods']['probe_all_layers']['auroc']:.3f} "
              f"diff={p['recall_diff_probe_minus_baseline']:+.3f} ci={p['ci95'][0]:+.3f}..{p['ci95'][1]:+.3f}")
    return 0


def cmd_fibs_check(a) -> int:
    from .capture import Capturer
    from . import fibs
    cap = Capturer(a.model, dtype=a.dtype)
    sample = fibs.parity_sample(a.parquet_dir, a.n_per_class, a.seed)
    res = fibs.parity_check(cap, sample, progress=lambda n, t: print(f"fibs {n}/{t}", file=sys.stderr, flush=True))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "fibs_parity.json").write_text(json.dumps(res, indent=2, default=float))
    write_manifest(out)
    print(json.dumps({k: round(v["auroc"], 3) for k, v in res["archs"].items()}), "rows", res["n_rows"])
    return 0


def cmd_fibs_eval(a) -> int:
    from . import pipeline
    return pipeline.cmd_fibs_eval(a)


def cmd_wsp_run(a) -> int:
    from . import pipeline
    return pipeline.cmd_wsp_run(a)


def cmd_final_test(a) -> int:
    from . import pipeline
    return pipeline.cmd_final_test(a)


def cmd_weights(a) -> int:
    from .weights_report import run_weight_report
    return run_weight_report(a)


def cmd_run_all(a) -> int:
    d, f, e, w = (str(Path(a.out) / x) for x in ("data", "features", "eval", "weights"))
    ns = argparse.Namespace
    cmd_build_data(ns(out=d, seed=a.seed, n_arith=a.n_arith, limit_facts=a.limit_facts, family_mix=True))
    cmd_capture(ns(model=a.model, data=d, out=f, dtype=a.dtype, skip_unbelieved=False, tokens=True))
    cmd_evaluate(ns(data=d, features=f, out=e, alpha=a.alpha, pooling="mean", n_boot=a.n_boot, seed=a.seed))
    cmd_weights(ns(model=a.model, data=d, features=f, out=w, dtype=a.weights_dtype, layer_pos=a.layer_pos, seed=a.seed,
                   n_null=a.n_null, alpha_edit=a.alpha_edit))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="liedet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-data"); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-arith", type=int, default=160); p.add_argument("--limit-facts", type=int, default=0)
    p.set_defaults(fn=cmd_build_data, family_mix=True)
    p = sub.add_parser("capture"); p.add_argument("--model", required=True); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--skip-unbelieved", action="store_true"); p.add_argument("--tokens", action="store_true")
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("evaluate"); p.add_argument("--data", required=True); p.add_argument("--features", required=True)
    p.add_argument("--out", required=True); p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--pooling", default="mean", choices=["mean", "last"]); p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("fibs-check"); p.add_argument("--model", required=True); p.add_argument("--parquet-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--dtype", default="bfloat16"); p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_fibs_check)
    p = sub.add_parser("fibs-eval"); p.add_argument("--data", required=True); p.add_argument("--features", required=True)
    p.add_argument("--out", required=True); p.add_argument("--variant", default="followup"); p.set_defaults(fn=cmd_fibs_eval)
    p = sub.add_parser("wsp-run"); p.add_argument("--data", required=True); p.add_argument("--features", required=True)
    p.add_argument("--out", required=True); p.add_argument("--variant", default="followup"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=6); p.add_argument("--per-round", type=int, default=5)
    p.add_argument("--proposer", choices=["claude", "random"], default="claude"); p.add_argument("--proposer-model", default="claude-opus-5-5")
    p.set_defaults(fn=cmd_wsp_run)
    p = sub.add_parser("final-test"); p.add_argument("--data", required=True); p.add_argument("--features", required=True)
    p.add_argument("--arms", required=True); p.add_argument("--out", required=True); p.add_argument("--variant", default="followup")
    p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_final_test)
    p = sub.add_parser("weights"); p.add_argument("--model", required=True); p.add_argument("--data", required=True)
    p.add_argument("--features", required=True); p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="float32"); p.add_argument("--layer-pos", type=int, default=3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--n-null", type=int, default=256)
    p.add_argument("--alpha-edit", type=float, default=1.0); p.set_defaults(fn=cmd_weights)
    p = sub.add_parser("run-all"); p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-arith", type=int, default=160); p.add_argument("--limit-facts", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--layer-pos", type=int, default=3); p.add_argument("--n-null", type=int, default=256)
    p.add_argument("--alpha-edit", type=float, default=1.0); p.add_argument("--weights-dtype", default="float32")
    p.set_defaults(fn=cmd_run_all)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
