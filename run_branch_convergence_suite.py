#!/usr/bin/env python3
"""Run the three remaining paper-fit convergence completions sequentially."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKER = BASE_DIR / "complete_branch_convergence.py"
STATUS_ROOT = (
    BASE_DIR
    / "outputs"
    / "paper_fit_suite_PCvariCepheidOnly_include2MASS_H_steps10000"
    / "convergence_completion_three_branches"
)
BRANCHES = ("fixed_g23", "wc2019", "f19")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_status() -> dict:
    path = STATUS_ROOT / "suite_status.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "pending",
        "created_at": now_iso(),
        "branches": {branch: {"status": "pending"} for branch in BRANCHES},
    }


def save_status(status: dict) -> None:
    status["updated_at"] = now_iso()
    atomic_json(STATUS_ROOT / "suite_status.json", status)


def run_command(branch: str, mode: str, workers: int) -> None:
    command = [
        sys.executable,
        str(WORKER),
        "--branch",
        branch,
        "--mode",
        mode,
        "--workers",
        str(workers),
    ]
    subprocess.run(command, check=True)


def run_suite(workers: int) -> None:
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    status = load_status()
    status["status"] = "running"
    status["pid"] = os.getpid()
    status.setdefault("started_at", now_iso())
    save_status(status)

    for branch in BRANCHES:
        branch_status = status["branches"].setdefault(branch, {})
        if branch_status.get("status") == "completed":
            continue
        branch_status.update({"status": "smoke_running", "started_at": now_iso()})
        save_status(status)
        run_command(branch, "smoke", workers)
        branch_status.update({"status": "full_running", "smoke_completed_at": now_iso()})
        save_status(status)
        run_command(branch, "full", workers)
        branch_status.update({"status": "completed", "completed_at": now_iso()})
        save_status(status)

    status.update({"status": "completed", "completed_at": now_iso()})
    save_status(status)


@contextlib.contextmanager
def suite_lock():
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    path = STATUS_ROOT / "suite.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another convergence suite holds {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={now_iso()}\n")
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    try:
        with suite_lock():
            run_suite(args.workers)
    except Exception as exc:
        status = load_status()
        status.update(
            {
                "status": "failed",
                "failed_at": now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        save_status(status)
        raise


if __name__ == "__main__":
    main()
