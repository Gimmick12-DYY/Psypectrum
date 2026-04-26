"""
Sequential experiment driver. Runs each named config in its own subprocess so a
crash on one run doesn't kill the rest, and CUDA / Python memory is released
between runs. Resumable: skips runs whose ``results/<name>/summary.json`` exists.

Usage:
    python run_experiments.py                     # all 10 in order
    python run_experiments.py --only 04 07        # only matching names
    python run_experiments.py --skip 09 10        # everything except matches
    python run_experiments.py --smoke             # tiny smoke run of every config
    python run_experiments.py --force             # re-run completed configs
    python run_experiments.py --no-compare        # skip viz_compare at the end
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

from experiments_config import EXPERIMENTS, all_names


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(p in name for p in patterns)


def select_runs(only: List[str], skip: List[str]) -> List[str]:
    names = all_names()
    if only:
        names = [n for n in names if _matches(n, only)]
    if skip:
        names = [n for n in names if not _matches(n, skip)]
    return names


def run_one(
    name: str,
    smoke: bool,
    force: bool,
    save_checkpoints: bool,
    device: str,
) -> int:
    effective_name = f"smoke_{name}" if smoke else name
    summary_path = RESULTS_DIR / effective_name / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {effective_name} already complete ({summary_path})")
        return 0

    log_dir = LOGS_DIR / effective_name
    log_dir.mkdir(parents=True, exist_ok=True)
    driver_log = log_dir / "driver.log"

    cmd = [sys.executable, "-u", "train_one.py", "--config", name, "--device", device]
    if smoke:
        cmd.append("--smoke")
    if force:
        cmd.append("--force")
    if save_checkpoints:
        cmd.append("--save-checkpoints")

    print(f"\n========== {effective_name} ==========")
    print(f"cmd: {' '.join(cmd)}")
    print(f"log: {driver_log}")
    t0 = time.time()
    rc = 0
    with open(driver_log, "w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
            text=True,
        )
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                sys.stdout.write(f"[{effective_name}] {line}")
                sys.stdout.flush()
                logf.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print(f"\n[!] interrupted during {effective_name}")
            raise
        rc = proc.wait()
    dt = time.time() - t0
    print(f"[done] {effective_name} rc={rc} in {dt:.1f}s")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[], help="Substrings: only matching names run")
    parser.add_argument("--skip", nargs="*", default=[], help="Substrings: matching names skipped")
    parser.add_argument("--smoke", action="store_true", help="Tiny dataset, 1 epoch each phase")
    parser.add_argument("--force", action="store_true", help="Re-run already-complete configs")
    parser.add_argument("--no-compare", action="store_true", help="Skip viz_compare at the end")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save best.pt / last.pt per run")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    names = select_runs(args.only, args.skip)
    if not names:
        print(f"No runs matched. Known: {all_names()}")
        return 1

    print(f"Running {len(names)} experiment(s):")
    for n in names:
        print(f"  - {n}")
    print()

    overall_t0 = time.time()
    failed: List[str] = []
    for name in names:
        try:
            rc = run_one(
                name,
                smoke=args.smoke,
                force=args.force,
                save_checkpoints=args.save_checkpoints,
                device=args.device,
            )
        except KeyboardInterrupt:
            return 130
        if rc != 0:
            failed.append(name)
    overall_dt = time.time() - overall_t0
    print(f"\n=== sweep done in {overall_dt / 60:.1f} min; failures: {failed or 'none'} ===")

    if not args.no_compare:
        print("\n=== generating comparison plots ===")
        try:
            cmp_cmd = [sys.executable, "-u", "viz_compare.py"]
            if args.smoke:
                cmp_cmd.append("--smoke")
            subprocess.run(cmp_cmd, check=False, cwd=str(PROJECT_ROOT))
        except Exception as e:
            print(f"viz_compare failed: {e}")

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
