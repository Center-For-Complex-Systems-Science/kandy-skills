#!/usr/bin/env python3
"""Run every KANDy example and aggregate its machine-checkable result.

Each example writes ``results/<Task>/metrics.json`` containing its metrics, the
thresholds it was judged against, and a ``passed`` boolean.  This runner
executes them, collects those files, and reports a single table.

    python run_all.py                     # everything
    python run_all.py odes maps           # only those categories
    python run_all.py --list              # show tasks and exit
    python run_all.py --timeout 1800      # per-example seconds (default 1800)

Environment note: run against the INSTALLED kandy package.  Do not point
PYTHONPATH at ~/Desktop/kandy/codebase/kandy/src — that source tree predates
``get_formula(lib=...)`` and fails 21 of the examples.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Scripts that only draw figures from another example's work — not discovery
# tasks, so they are skipped.  None at present.
NON_TASK_SCRIPTS: set[str] = set()


def discover(categories: list[str] | None) -> list[Path]:
    scripts = sorted(
        p for p in HERE.glob("*/*.py")
        if not p.name.startswith("_") and p.name not in NON_TASK_SCRIPTS
    )
    if categories:
        wanted = set(categories)
        scripts = [p for p in scripts if p.parent.name in wanted]
    return scripts


def _norm(s: str) -> str:
    return s.lower().replace("_example", "").replace("_", "").replace("-", "")


def _matches(task_name: str, script: Path) -> bool:
    """Does this metrics 'task' field belong to this script?"""
    a, b = _norm(task_name), _norm(script.stem)
    return a == b or a in b or b in a


def _task_name_of(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("task")
    except (json.JSONDecodeError, OSError):
        return None


def metrics_mtimes() -> dict[Path, float]:
    return {p: p.stat().st_mtime for p in RESULTS.glob("*/metrics.json")}


def run_one(script: Path, timeout: int) -> dict:
    """Execute one example; return a record of what happened."""
    rel = script.relative_to(HERE)
    before = metrics_mtimes()
    t0 = time.time()
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kandy")

    try:
        proc = subprocess.run(
            [sys.executable, "-W", "ignore", str(rel)],
            cwd=HERE, env=env, timeout=timeout,
            capture_output=True, text=True,
        )
        status = "ok" if proc.returncode == 0 else "error"
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
        detail = "" if status == "ok" else tail[0][:160]
    except subprocess.TimeoutExpired:
        status, detail = "timeout", f"exceeded {timeout}s"

    runtime = time.time() - t0

    # Find the metrics.json this run produced (new file, or freshly rewritten).
    after = metrics_mtimes()
    written = [p for p, m in after.items() if m > before.get(p, 0.0)]
    record = {
        "script": str(rel), "category": script.parent.name,
        "status": status, "detail": detail, "runtime_s": round(runtime, 1),
        "metrics_written": bool(written), "passed": None, "task": script.stem,
    }
    # Attribute by NAME, not merely by "written most recently".  Anything else
    # running concurrently — including the user invoking an example directly —
    # also touches results/, and mtime alone will happily credit this task with
    # someone else's file (observed: a Burgers-Fourier run reported as
    # "HeatFluxClosure").  Prefer a written file whose task name matches this
    # script; fall back to mtime only when exactly one file changed.
    matched = [p for p in written
               if _matches(_task_name_of(p) or p.parent.name, script)]
    chosen = matched[0] if matched else (written[0] if len(written) == 1 else None)
    if chosen is not None:
        try:
            data = json.loads(chosen.read_text())
            record["passed"] = data.get("passed")
            record["task"] = data.get("task", chosen.parent.name)
            record["metrics_path"] = str(chosen.relative_to(HERE))
            if not matched:
                record["detail"] = ("attributed by mtime only — name did not "
                                    "match; check for concurrent runs")
        except (json.JSONDecodeError, OSError) as exc:
            record["detail"] = f"unreadable metrics.json: {exc}"
    elif written:
        record["detail"] = (f"{len(written)} metrics files changed and none "
                            f"matched this task — concurrent run?")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="*",
                    help="odes maps oscillators pdes fluids plasma mathbio geometry")
    ap.add_argument("--timeout", type=int, default=1800, help="per-example seconds")
    ap.add_argument("--list", action="store_true", help="list tasks and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip tasks whose metrics.json is newer than their script")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these scripts (paths relative to examples/)")
    args = ap.parse_args()

    # Metrics files are attributed to a task by mtime, so two runners at once
    # will steal each other's results. Refuse rather than report nonsense.
    lock = RESULTS / ".run_all.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not args.list:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            other = lock.read_text().strip()
            print(f"another run_all.py is active (pid {other}); "
                  f"remove {lock} if that is stale", file=sys.stderr)
            return 2

    scripts = discover(args.categories)
    if args.list:
        for s in scripts:
            print(s.relative_to(HERE))
        print(f"\n{len(scripts)} tasks")
        return 0

    if args.only:
        wanted = {str(Path(o)) for o in args.only}
        scripts = [s for s in scripts if str(s.relative_to(HERE)) in wanted]

    if args.resume:
        # A task is up to date when the metrics.json it produced is newer than
        # the script itself, so editing one example re-queues exactly that one.
        # The script names its own output dir via the "task" field, so read it
        # rather than guessing from the filename.
        owner = {}
        for m in RESULTS.glob("*/metrics.json"):
            try:
                owner[json.loads(m.read_text()).get("task", m.parent.name)] = m
            except (json.JSONDecodeError, OSError):
                continue
        skipped, remaining = [], []
        for s in scripts:
            m = next((mm for t, mm in owner.items()
                      if _matches(t, s) and mm.stat().st_mtime > s.stat().st_mtime),
                     None)
            (skipped if m else remaining).append(s)
        if skipped:
            print(f"--resume: skipping {len(skipped)} task(s) already up to date")
        scripts = remaining

    print(f"Running {len(scripts)} examples (timeout {args.timeout}s each)\n")
    records = []
    for i, script in enumerate(scripts, 1):
        rel = script.relative_to(HERE)
        print(f"[{i:2d}/{len(scripts)}] {rel} ...", flush=True)
        rec = run_one(script, args.timeout)
        records.append(rec)
        verdict = {True: "PASS", False: "FAIL", None: "no metrics"}[rec["passed"]]
        print(f"          {rec['status']:8s} {verdict:10s} {rec['runtime_s']:7.1f}s"
              f"{'  ' + rec['detail'] if rec['detail'] else ''}", flush=True)

    # -- summary ------------------------------------------------------------
    n = len(records)
    passed = sum(r["passed"] is True for r in records)
    failed = sum(r["passed"] is False for r in records)
    unscored = sum(r["passed"] is None for r in records)
    crashed = sum(r["status"] != "ok" for r in records)

    print("\n" + "=" * 78)
    print(f"{'task':28s} {'category':11s} {'status':9s} {'verdict':10s} {'runtime':>9s}")
    print("-" * 78)
    for r in sorted(records, key=lambda r: (r["category"], r["task"])):
        verdict = {True: "PASS", False: "FAIL", None: "-"}[r["passed"]]
        print(f"{r['task'][:28]:28s} {r['category']:11s} {r['status']:9s} "
              f"{verdict:10s} {r['runtime_s']:8.1f}s")
    print("-" * 78)
    print(f"{n} tasks: {passed} pass, {failed} fail, {unscored} unscored, "
          f"{crashed} did not complete")
    print(f"total wall clock: {sum(r['runtime_s'] for r in records) / 60:.1f} min")

    summary = {
        "n_tasks": n, "n_passed": passed, "n_failed": failed,
        "n_unscored": unscored, "n_crashed": crashed,
        "total_runtime_s": round(sum(r["runtime_s"] for r in records), 1),
        "python": sys.version.split()[0],
        "records": records,
    }
    out = RESULTS / "benchmark_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out.relative_to(HERE)}")
    lock.unlink(missing_ok=True)
    return 0 if failed == 0 and crashed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
