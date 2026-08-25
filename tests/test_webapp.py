from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.app import create_app
from webapp.config import Settings
from webapp.security import hash_password


def csrf_from(response) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
    if not match:
        raise AssertionError("CSRF token was not rendered")
    return match.group(1)


class WebApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(self.temp_dir.name)
        settings = Settings(
            data_dir=data_dir,
            database_path=data_dir / "test.sqlite3",
            secret_key="test-secret-key-with-enough-entropy",
            allowed_hosts=["testserver"],
            secure_cookies=False,
            pipeline_python="python",
        )
        self.app = create_app(settings)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.database = self.app.state.database

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def register(self, email: str, name: str = "Test Chemist", password: str = "correct horse battery staple"):
        page = self.client.get("/register")
        return self.client.post(
            "/register",
            data={
                "csrf_token": csrf_from(page),
                "display_name": name,
                "email": email,
                "password": password,
                "password_confirmation": password,
            },
            follow_redirects=False,
        )

    def login(self, client: TestClient, email: str, password: str):
        page = client.get("/login")
        return client.post(
            "/login",
            data={"csrf_token": csrf_from(page), "email": email, "password": password},
            follow_redirects=False,
        )

    def create_approved_user(self, email: str, password: str, role: str = "user"):
        return self.database.create_user(
            email=email,
            display_name=email.split("@", 1)[0].title(),
            password_hash=hash_password(password),
            role=role,
            status="approved",
        )

    def test_registration_requires_staff_approval(self) -> None:
        response = self.register("chemist@example.com")
        self.assertEqual(response.status_code, 303)
        user = self.database.get_user_by_email("chemist@example.com")
        self.assertEqual(user["status"], "pending")

        login = self.login(self.client, "chemist@example.com", "correct horse battery staple")
        self.assertEqual(login.status_code, 403)
        self.assertIn("waiting for approval", login.text)

    def test_development_admin_can_login_with_default_credentials(self) -> None:
        login = self.login(self.client, "admin", "admin")
        self.assertEqual(login.status_code, 303)
        self.assertEqual(login.headers["location"], "/dashboard")
        dashboard = self.client.get("/dashboard")
        self.assertIn("Platform administrator", dashboard.text)

    def test_admin_can_approve_and_user_can_create_private_job(self) -> None:
        password = "correct horse battery staple"
        self.register("scientist@example.com", password=password)
        pending = self.database.get_user_by_email("scientist@example.com")
        self.create_approved_user("admin@example.com", password, role="admin")

        admin_client = TestClient(self.app)
        self.addCleanup(admin_client.close)
        login = self.login(admin_client, "admin@example.com", password)
        self.assertEqual(login.status_code, 303)
        approvals = admin_client.get("/admin/users")
        self.assertIn("scientist@example.com", approvals.text)
        approved = admin_client.post(
            f"/admin/users/{pending['id']}/review",
            data={"csrf_token": csrf_from(approvals), "status": "approved", "role": "user"},
            follow_redirects=False,
        )
        self.assertEqual(approved.status_code, 303)
        self.assertEqual(self.database.get_user(pending["id"])["status"], "approved")

        user_client = TestClient(self.app)
        self.addCleanup(user_client.close)
        login = self.login(user_client, "scientist@example.com", password)
        self.assertEqual(login.status_code, 303)
        form = user_client.get("/jobs/new")
        response = user_client.post(
            "/jobs",
            data={
                "csrf_token": csrf_from(form),
                "name": "August screen",
                "preset": "docking",
                "ionise": "1",
                "conformers": "1",
                "n_conformers": "1",
                "max_unspecified_stereo": "2",
                "max_tautomers": "5",
                "ph_min": "7.4",
                "ph_max": "7.4",
                "chunk_size": "100000",
                "batch_size": "500",
                "batches_per_gpu": "4",
                "preprocessing_threads": "8",
                "mmff_max_iters": "200",
                "pains_backend": "auto",
            },
            files={"files": ("sample.smi", b"CCO ethanol\n", "text/plain")},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        job = self.database.get_job(job_id)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["user_id"], pending["id"])

        stranger_password = "another safe password value"
        self.create_approved_user("stranger@example.com", stranger_password)
        stranger = TestClient(self.app)
        self.addCleanup(stranger.close)
        self.login(stranger, "stranger@example.com", stranger_password)
        denied = stranger.get(f"/jobs/{job_id}", follow_redirects=False)
        self.assertEqual(denied.status_code, 403)

        detail = user_client.get(f"/jobs/{job_id}")
        canceled = user_client.post(
            f"/jobs/{job_id}/cancel",
            data={"csrf_token": csrf_from(detail)},
            follow_redirects=False,
        )
        self.assertEqual(canceled.status_code, 303)
        self.assertEqual(self.database.get_job(job_id)["status"], "canceled")

    def test_csrf_protects_registration(self) -> None:
        response = self.client.post(
            "/register",
            data={
                "csrf_token": "invalid",
                "display_name": "Bad Request",
                "email": "bad@example.com",
                "password": "correct horse battery staple",
                "password_confirmation": "correct horse battery staple",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(self.database.get_user_by_email("bad@example.com"))

    def test_persistent_rate_limit_can_be_cleared_after_success(self) -> None:
        key = "test-login-key"
        self.assertTrue(self.database.consume_rate_limit(key, limit=2, window_seconds=900))
        self.assertTrue(self.database.consume_rate_limit(key, limit=2, window_seconds=900))
        self.assertFalse(self.database.consume_rate_limit(key, limit=2, window_seconds=900))
        self.database.clear_rate_limit(key)
        self.assertTrue(self.database.consume_rate_limit(key, limit=2, window_seconds=900))

    def test_root_path_is_used_for_links_and_redirects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            app = create_app(
                Settings(
                    root_path="/tools/libprep",
                    data_dir=data_dir,
                    database_path=data_dir / "prefixed.sqlite3",
                    secret_key="test-secret-key-with-enough-entropy",
                    allowed_hosts=["testserver"],
                    secure_cookies=False,
                )
            )
            with TestClient(app) as client:
                landing = client.get("/")
                self.assertIn('href="/tools/libprep/login"', landing.text)
                self.assertIn('href="http://testserver/tools/libprep/static/app.css"', landing.text)
                login = client.get("/login")
                self.assertIn("Path=/tools/libprep/", login.headers["set-cookie"])
                redirect = client.get("/dashboard", follow_redirects=False)
                self.assertEqual(redirect.headers["location"], "/tools/libprep/login")


if __name__ == "__main__":
    unittest.main()
