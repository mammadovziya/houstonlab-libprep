from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from webapp.config import Settings
from webapp.worker import Worker


class FinishedProcess:
    def __init__(self, return_code: int, output: str = ""):
        self.return_code = return_code
        self.stdout = io.StringIO(output)

    def poll(self) -> int:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        return self.return_code


class CancelingProcess(FinishedProcess):
    def __init__(self):
        super().__init__(-15)
        self.poll_count = 0

    def poll(self) -> int | None:
        self.poll_count += 1
        return None if self.poll_count == 1 else self.return_code


class WorkerOutputRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(self.temp_dir.name)
        self.worker = Worker(
            Settings(
                data_dir=data_dir,
                database_path=data_dir / "test.sqlite3",
                secret_key="test-secret-key-with-enough-entropy",
                allowed_hosts=["testserver"],
                secure_cookies=False,
                pipeline_python="python",
            )
        )
        user = self.worker.database.create_user(
            email="worker@example.com",
            display_name="Worker Test",
            password_hash="unused",
            status="approved",
        )
        self.job_id = "worker-output-test"
        self.worker.database.create_job(
            job_id=self.job_id,
            user_id=user["id"],
            name="Worker output test",
            preset="docking",
            params={},
            files=[],
        )
        self.job = self.worker.database.get_job(self.job_id)
        self.output_dir = data_dir / "jobs" / self.job_id / "output"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_with_process(self, process_factory) -> None:
        with (
            mock.patch.object(self.worker, "command_for", return_value=["fake-pipeline"]),
            mock.patch("webapp.worker.subprocess.Popen", side_effect=process_factory),
        ):
            self.worker.run_job(self.job)

    def output_files(self) -> list[dict]:
        return [
            file
            for file in self.worker.database.job_files(self.job_id)
            if file["kind"] == "output"
        ]

    def test_failed_job_registers_only_downloadable_diagnostics(self) -> None:
        artifacts = {
            "library_conformer_failures.csv": "ID,reason\nmol-1,embedding failed\n",
            "library_mmff_unparametrizable.csv": "ID,reason\nmol-2,no parameters\n",
            "library.sdf": "partial sdf",
            "library_final.smi": "CCO mol-1\n",
            "library_metadata.csv": "ID\nmol-1\n",
            "other.csv": "partial,product\n",
        }

        def failed_process(*args, **kwargs):
            for filename, contents in artifacts.items():
                (self.output_dir / filename).write_text(contents, encoding="utf-8")
            return FinishedProcess(1, "fatal pipeline error\n")

        self.run_with_process(failed_process)

        output_files = self.output_files()
        registered_names = {file["filename"] for file in output_files}
        expected_names = {
            "pipeline.log",
            "library_conformer_failures.csv",
            "library_mmff_unparametrizable.csv",
        }
        self.assertEqual(registered_names, expected_names)
        expected_bytes = sum(
            (self.output_dir / filename).stat().st_size for filename in expected_names
        )
        job = self.worker.database.get_job(self.job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["output_bytes"], expected_bytes)
        self.assertEqual(sum(file["size_bytes"] for file in output_files), expected_bytes)
        self.assertIn("fatal pipeline error", job["error_message"])

    def test_successful_job_still_registers_all_regular_outputs(self) -> None:
        artifacts = {
            "library.sdf": "complete sdf",
            "library_final.smi": "CCO mol-1\n",
            "library_metadata.csv": "ID\nmol-1\n",
            ".libprep.pipeline.lock": "",
        }

        def successful_process(*args, **kwargs):
            for filename, contents in artifacts.items():
                (self.output_dir / filename).write_text(contents, encoding="utf-8")
            return FinishedProcess(0, "PIPELINE COMPLETE\n")

        self.run_with_process(successful_process)

        output_files = self.output_files()
        expected_names = {
            "pipeline.log",
            *(name for name in artifacts if not name.endswith(".lock")),
        }
        self.assertEqual({file["filename"] for file in output_files}, expected_names)
        expected_bytes = sum(
            (self.output_dir / filename).stat().st_size for filename in expected_names
        )
        job = self.worker.database.get_job(self.job_id)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["output_bytes"], expected_bytes)

    def test_canceled_job_does_not_register_partial_outputs(self) -> None:
        self.worker.database.update_job(self.job_id, status="running")
        self.worker.database.request_cancel(self.job_id)

        def canceled_process(*args, **kwargs):
            (self.output_dir / "library_conformer_failures.csv").write_text(
                "ID,reason\nmol-1,canceled\n", encoding="utf-8"
            )
            (self.output_dir / "library.sdf").write_text("partial sdf", encoding="utf-8")
            return CancelingProcess()

        with mock.patch("webapp.worker.select.select", return_value=([], [], [])):
            self.run_with_process(canceled_process)

        job = self.worker.database.get_job(self.job_id)
        self.assertEqual(job["status"], "canceled")
        self.assertEqual(job["output_bytes"], 0)
        self.assertEqual(self.output_files(), [])


if __name__ == "__main__":
    unittest.main()
