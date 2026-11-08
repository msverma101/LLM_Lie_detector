"""Apply the new form checker to actual old artifacts without rewriting results."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous-run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scripts = Path(__file__).resolve().parents[2] / ".agents/skills/detector-weight-space-debugging/scripts"
    spec = importlib.util.spec_from_file_location("audit_edit", scripts / "audit_edit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    root = Path(args.previous_run)
    results = []
    for folder in sorted(root.glob("attempt_*")):
        if not folder.is_dir():
            continue
        proposal = json.loads((folder / "spec.json").read_text())
        parent = root / proposal["parent"]
        result = audit.inspect(audit.load(parent), audit.load(folder / "wsp.npz"))
        result.update(attempt=proposal["attempt"], parent=proposal["parent"],
                      original_decision=json.loads((folder / "feedback.json").read_text())["kept"])
        results.append(result)
    out = {"scope": "retrospective form/integrity check, not new efficacy evaluation",
           "attempts": results, "rejected_as_mechanistic": [r["attempt"] for r in results if not r["allowed"]]}
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps({"attempts": len(results), "rejected_as_mechanistic": out["rejected_as_mechanistic"]}))


if __name__ == "__main__":
    main()
