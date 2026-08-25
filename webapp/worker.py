from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import select
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from webapp.config import PROJECT_DIR, Settings
from webapp.db import Database, utcnow


STAGE_MARKERS = {
    "STEP 1:": "Loading supplier files",
    "STEP 2:": "Stripping salts",
    "STEP 3:": "Filtering compounds",
    "STEP 4:": "Enumerating stereochemistry",
    "STEP 4b:": "Enumerating tautomers",
    "STEP 5:": "Deduplicating structures",
    "STEP 6:": "Assigning protonation states",
    "STEP 7:": "Generating 3D conformers",
    "PIPELINE COMPLETE": "Packaging results",
}

FAILURE_DIAGNOSTIC_SUFFIXES = (
    "_conformer_failures.csv",
    "_mmff_unparametrizable.csv",
)


def file_digest(path: Path) -> str:
    if path.stat().st_size > 2 * 1024**3:
        return "not-computed-large-file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Worker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = Database(settings.database_path)
        self.database.initialize()
        self.last_heartbeat = 0.0

    def heartbeat(self, value: str = "idle") -> None:
        now = time.monotonic()
        if now - self.last_heartbeat >= 5:
            self.database.set_service_state("worker", value)
            self.last_heartbeat = now

    def command_for(self, job: dict[str, Any], files: list[dict[str, Any]], output: Path) -> list[str]:
        inputs = [file["stored_path"] for file in files if file["kind"] == "input"]
        custom = next((file["stored_path"] for file in files if file["kind"] == "custom_smarts"), None)
        params = job["params"]
        command = [
            self.settings.pipeline_python,
            str(PROJECT_DIR / "library_pipeline.py"),
            "--input", *inputs,
            "--output", str(output),
            "--preset", job["preset"],
            "--n-conformers", str(params["n_conformers"]),
            "--max-unspecified-stereo", str(params["max_unspecified_stereo"]),
            "--max-tautomers", str(params["max_tautomers"]),
            "--ph-min", str(params["ph_min"]),
            "--ph-max", str(params["ph_max"]),
            "--chunk-size", str(params["chunk_size"]),
            "--batch-size", str(params["batch_size"]),
            "--batches-per-gpu", str(params["batches_per_gpu"]),
            "--preprocessing-threads", str(params["preprocessing_threads"]),
            "--mmff-max-iters", str(params["mmff_max_iters"]),
            "--pains-backend", params["pains_backend"],
        ]
        command.append("--tautomers" if params["tautomers"] else "--skip-tautomers")
        if not params["ionise"]:
            command.append("--skip-ionise")
        if not params["conformers"]:
            command.append("--skip-conformers")
        if params["save_intermediates"]:
            command.append("--save-intermediates")
        if custom:
            command.extend(("--custom-smarts", custom))
        return command

    def update_from_line(self, job_id: str, line: str) -> None:
        stripped = line.strip()
        for marker, stage in STAGE_MARKERS.items():
            if marker in stripped:
                self.database.update_job(job_id, stage=stage, progress_message=stripped[:300])
                return
        if stripped.startswith("Chunk ") or "  Chunk " in line:
            self.database.update_job(job_id, stage="Generating 3D conformers", progress_message=stripped[:300])
        elif stripped.startswith(("Final SMILES:", "Final metadata:")):
            self.database.update_job(job_id, progress_message=stripped[:300])

    def terminate(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def register_output_files(
        self, job_id: str, output_dir: Path, *, diagnostics_only: bool = False
    ) -> int:
        output_bytes = 0
        for path in sorted(output_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith(".") and path.name.endswith(".lock"):
                continue
            if diagnostics_only and not (
                path.name == "pipeline.log"
                or path.name.endswith(FAILURE_DIAGNOSTIC_SUFFIXES)
            ):
                continue
            size = path.stat().st_size
            output_bytes += size
            self.database.add_output_file(
                job_id, path.name, str(path.resolve()), size, file_digest(path)
            )
        return output_bytes

    def run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        files = self.database.job_files(job_id)
        job_dir = self.settings.data_dir / "jobs" / job_id
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_sdf = output_dir / "library.sdf"
        log_path = output_dir / "pipeline.log"
        command = self.command_for(job, files, output_sdf)
        self.database.update_job(
            job_id,
            log_path=str(log_path.resolve()),
            stage="Starting pipeline",
            progress_message="Launching the preparation process",
        )
        tail: list[str] = []
        canceled = False
        return_code = -1
        process: subprocess.Popen[str] | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                assert process.stdout is not None
                while process.poll() is None:
                    self.heartbeat(f"running:{job_id}")
                    readable, _, _ = select.select([process.stdout], [], [], 1.0)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            log.write(line)
                            log.flush()
                            tail.append(line.rstrip())
                            tail = tail[-40:]
                            self.update_from_line(job_id, line)
                    if self.database.is_cancel_requested(job_id):
                        canceled = True
                        self.terminate(process)
                        break
                for line in process.stdout:
                    log.write(line)
                    tail.append(line.rstrip())
                    tail = tail[-40:]
                return_code = process.wait()
        except Exception as exc:
            if process is not None:
                self.terminate(process)
                return_code = process.poll() if process.poll() is not None else -1
            tail.append(f"Worker error: {exc}")

        if canceled:
            self.database.update_job(
                job_id, status="canceled", stage="Canceled",
                progress_message="The job was canceled", return_code=return_code,
                finished_at=utcnow(),
            )
            return

        if return_code == 0:
            output_bytes = self.register_output_files(job_id, output_dir)
            self.database.update_job(
                job_id, status="succeeded", stage="Complete",
                progress_message="Results are ready to download", output_bytes=output_bytes,
                return_code=return_code, finished_at=utcnow(),
            )
        else:
            output_bytes = self.register_output_files(
                job_id, output_dir, diagnostics_only=True
            )
            message = "\n".join(tail)[-6000:] or "Pipeline process exited without an error message"
            self.database.update_job(
                job_id, status="failed", stage="Failed",
                progress_message="The pipeline stopped before completion", return_code=return_code,
                error_message=message, output_bytes=output_bytes, finished_at=utcnow(),
            )

    def run_forever(self, poll_seconds: float = 2.0, once: bool = False) -> None:
        lock_path = self.settings.data_dir / "worker.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SystemExit("Another LibPrep GPU worker is already running") from exc
            self.database.recover_interrupted_jobs()
            while True:
                self.heartbeat("idle")
                job = self.database.claim_next_job()
                if job:
                    self.run_job(job)
                    self.last_heartbeat = 0
                if once:
                    break
                time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="LibPrep persistent GPU job worker")
    parser.add_argument("--once", action="store_true", help="Run at most one queued job")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    Worker(Settings()).run_forever(max(args.poll_seconds, 0.25), once=args.once)


if __name__ == "__main__":
    main()
