#!/usr/bin/env python3
"""Verifier stub for the TAPT-Caltech101 task (v0.1 draft).

Scores the auto-checkable leaves of tests/rubrics.json against the agent's
/workspace/submission. Leaves whose checks are not yet implementable (oracle
diff, human-review items) are skipped and reported. Writes the reward for the
platform at /logs/verifier/reward.txt (0.0-1.0) and a JSON detail dump.

This is a DRAFT boundary: full rubric reward requires the human-reviewed
rubrics.json and the locked oracle tolerances (see REVISION_NOTES.md).
"""
import json
import pathlib
import sys

SUB = pathlib.Path("/workspace/submission")
RUB = pathlib.Path("/workspace/tests/rubrics.json")


def load():
    return json.loads(RUB.read_text()) if RUB.exists() else None


def check_layout(rub) -> dict:
    ok = {}
    ok["layout.submission_dir"] = bool((SUB / "evaluation").is_dir())
    ok["layout.metrics_files"] = bool(
        (SUB / "metrics_adv_ivlp.json").is_file()
        and (SUB / "metrics_clean_ivlp.json").is_file()
    )
    ok["layout.report_md"] = bool(
        (SUB / "REPORT.md").is_file() and (SUB / "REPORT.md").stat().st_size > 0
    )
    return ok


def check_schema(rub) -> dict:
    REQUIRED = {
        "dataset", "method", "checkpoint", "checkpoint_name", "clean_top1",
        "robust_top1", "tap_before_top1", "tap_after_top1", "attack", "eps", "steps",
    }
    ok = {}
    paths = [SUB / "metrics_adv_ivlp.json", SUB / "metrics_clean_ivlp.json"]
    try:
        adv, clean = (json.loads(p.read_text()) for p in paths)
        ok["schema.json_parses"] = True
    except Exception:
        ok["schema.json_parses"] = False
        return ok
    objs = {"adv": adv, "clean": clean}
    ok["schema.keys_exact"] = all(REQUIRED <= set(o) for o in objs.values()) or (
        (REQUIRED - {"robust_top1"}) <= set(objs["clean"]) and "robust_top1" not in objs["clean"]
        and REQUIRED <= set(objs["adv"])
    )
    ok["schema.types"] = all(
        isinstance(o[k], (int, float)) and 0 <= o[k] <= 100
        for o in objs.values() for k in ("clean_top1", "tap_before_top1", "tap_after_top1")
    ) and all(
        isinstance(o.get("robust_top1"), (int, float)) and 0 <= o["robust_top1"] <= 100
        for o in objs.values() if o.get("robust_top1") is not None
    ) and all(o["attack"] == "auto" and o["eps"] == "1/255" and o["steps"] == 100 for o in objs.values())
    ok["schema.checkpoint_tag"] = objs["adv"]["checkpoint"] == "AdvIVLP" and objs["clean"]["checkpoint"] == "CleanIVLP"
    report = (SUB / "REPORT.md").read_text()
    ok["schema.determinism_field"] = ("pkl" in report.lower()) and ("bundled" in report.lower() or "generated" in report.lower())
    return ok


def check_science(rub) -> dict:
    ok = {}
    adv = json.loads((SUB / "metrics_adv_ivlp.json").read_text())
    clean = json.loads((SUB / "metrics_clean_ivlp.json").read_text())
    for o in (adv, clean):
        c, r, b, a = o["clean_top1"], o.get("robust_top1"), o["tap_before_top1"], o["tap_after_top1"]
        ok.setdefault("science.clean_top1_range", []).append(80 <= c <= 96 or (c != c))
        ok.setdefault("science.robust_top1_range", []).append(
            (r is not None and (40 <= r <= 86 or (r != r))) or r is None)
        ok.setdefault("science.clean_vs_robust", []).append(r is None or r <= c + 1e-6)
        ok.setdefault("science.before_vs_after", []).append(b <= a + 2.0)
    ok["science.clean_top1_range"] = all(ok["science.clean_top1_range"])
    ok["science.robust_top1_range"] = all(ok["science.robust_top1_range"])
    ok["science.clean_vs_robust"] = all(ok["science.clean_vs_robust"])
    ok["science.before_vs_after"] = all(ok["science.before_vs_after"])
    ok["science.match_oracle_clean"] = None  # pending locked oracle
    ok["science.match_oracle_robust"] = None
    return ok


def main():
    rerun_bash = None
    rub = load()
    if rub is None:
        print("no rubrics.json — cannot score", file=sys.stderr)
        sys.exit(1)
    results = {}
    results.update(check_layout(rub))
    results.update(check_schema(rub))
    results.update(check_science(rub))

    scored, passed = 0.0, 0.0
    detail = {"leaf_results": results, "skipped": [], "notes": []}
    for sec in rub["sections"]:
        for leaf in sec["leaves"]:
            lid = leaf["id"]
            r = results.get(lid)
            if r is None:
                detail["skipped"].append(lid)
                continue
            scored += leaf["weight"]
            if r is True:
                passed += leaf["weight"]
    reward = passed / scored if scored > 0 else 0.0
    detail["reward"] = reward
    detail["scored_weight"] = scored
    detail["passed_weight"] = passed

    out = pathlib.Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.txt").write_text(f"{reward:.6f}\n")
    (out / "detail.json").write_text(json.dumps(detail, indent=2))
    print(f"reward: {reward:.4f} (scored {scored:.2f}/{passed:.2f}, skipped {len(detail['skipped'])})")


if __name__ == "__main__":
    main()