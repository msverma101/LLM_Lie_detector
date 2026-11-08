"""Recompute WSP source-record counts, not model inference or author reasoning."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

REV = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == REV
    def read(name):
        return json.loads((root / name).read_text())
    prefix = "exp8_math/runs/p50c/"
    rows = [json.loads(x) for x in (root / (prefix + "tool_calls.jsonl")).read_text().splitlines()]
    forwards = [r for r in rows if r.get("kind") == "forward"]
    successful = [r for r in forwards if r.get("result", {}).get("ok")]
    state = read(prefix + "state.json")
    first = successful[0]["result"]
    manifest = read(prefix + "edit/manifest.json")
    # Confirmation CSV has per-item outcomes; inspect the header instead of assuming fields.
    with (root / (prefix + "work_notes/final_confirmation.csv")).open() as handle:
        confirmation = list(csv.DictReader(handle))
    benchmarks = {name: read("exp8_math/eval/" + name + "/merged.json") for name in ["unedited", "p50c"]}
    facts = {"source_repository": "https://github.com/RikiyaT/lm_understand-logs", "revision": REV,
             "verification_type": "recomputed released records; no independent inference",
             "python_runs": sum(r.get("kind") == "python" for r in rows), "turns": state["turn"],
             "forward_requests": len(forwards), "successful_passes": len(successful),
             "failed_requests": len(forwards) - len(successful),
             "successful_by_type": {t: sum(r["request"]["type"] == t for r in successful)
                                    for t in ["generate", "teacher_forced"]},
             "tool_log_span_hours": (rows[-1]["t"] - rows[0]["t"]) / 3600,
             "initial_generations": first["generations"], "initial_accuracy": first["accuracy"],
             "initial_failures": first["failures"],
             "touched_values_from_shapes": sum(t["shape"][0] * t["changed_columns"] for t in manifest["tensors"]),
             "changed_bfloat16_values": manifest["changed_entries_total"],
             "confirmation_rows": len(confirmation), "confirmation_columns": list(confirmation[0]),
             "confirmation": {"original_correct": sum(int(r["original"]) for r in confirmation),
                 "edited_correct": sum(int(r["edited"]) for r in confirmation),
                 "rescues": sum(int(r["original"]) == 0 and int(r["edited"]) == 1 for r in confirmation),
                 "regressions": sum(int(r["original"]) == 1 and int(r["edited"]) == 0 for r in confirmation),
                 "rescued_originally_cut": sum(int(r["original"]) == 0 and int(r["edited"]) == 1 and int(r["original_cut"]) == 1 for r in confirmation)},
             "benchmarks": {b: {n: benchmarks[n][b] for n in benchmarks} for b in ["math500", "gsm8k", "ifeval"]}}
    assert facts["python_runs"] == state["runs"] == 80
    assert facts["successful_passes"] == state["passes"] == 50
    assert facts["touched_values_from_shapes"] == 131072
    assert facts["changed_bfloat16_values"] == 130188
    files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    facts["consulted_files"] = [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size, "url": "https://github.com/RikiyaT/lm_understand-logs/blob/" + REV + "/" + str(p.relative_to(root))}
        for p in sorted(files)]
    (out / "SOURCE_AUDIT.json").write_text(json.dumps(facts, indent=2, allow_nan=False))
    with (out / "ORIGINAL_FORWARD_LEDGER.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["request_id", "type", "model", "ok", "n_items", "error"])
        writer.writeheader()
        for row in forwards:
            req, result = row["request"], row["result"]
            writer.writerow({"request_id": row["k"], "type": req.get("type"), "model": req.get("model"),
                "ok": bool(result.get("ok")), "n_items": len(req.get("problems", req.get("texts", req.get("chats", [])))),
                "error": result.get("error", "")})
    # All Python blocks are indexed, including failed or reverted interventions.
    with (out / "ORIGINAL_PYTHON_LEDGER.jsonl").open("w") as handle:
        for i, row in enumerate(rows):
            if row.get("kind") == "python":
                code = row["code"]
                handle.write(json.dumps({"tool_log_row": i + 1, "timestamp": row["t"], "seconds": row.get("s"),
                    "hypothesis_comment": next((x for x in code.splitlines() if x.strip()), ""),
                    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "mentions_save_edit": "save_edit(" in code,
                    "note": "mention may be a helper definition, not a distinct applied candidate"}) + "\n")
    print(json.dumps({k: v for k, v in facts.items() if k not in ["consulted_files", "benchmarks"]}, indent=2))


if __name__ == "__main__":
    main()
