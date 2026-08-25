from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from webapp.security import token_hash


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'moderator', 'user')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'disabled')),
    reviewed_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at TEXT,
    last_login_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_status_created
ON users(status, created_at);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf_token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'canceled')),
    preset TEXT NOT NULL CHECK (preset IN ('docking', 'enumerate')),
    params_json TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'Queued',
    progress_message TEXT NOT NULL DEFAULT 'Waiting for the GPU worker',
    input_count INTEGER NOT NULL DEFAULT 0,
    input_bytes INTEGER NOT NULL DEFAULT 0,
    output_bytes INTEGER NOT NULL DEFAULT 0,
    return_code INTEGER,
    error_message TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
    log_path TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS job_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('input', 'custom_smarts', 'output')),
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_files_job_kind ON job_files(job_id, kind);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    ip_address TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);

CREATE TABLE IF NOT EXISTS service_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limits (
    key_hash TEXT PRIMARY KEY,
    attempts INTEGER NOT NULL,
    window_started REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def connection(self):
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA optimize")

    def create_user(
        self,
        *,
        email: str,
        display_name: str,
        password_hash: str,
        role: str = "user",
        status: str = "pending",
    ) -> dict[str, Any]:
        user_id = uuid.uuid4().hex
        now = utcnow()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO users
                    (id, email, display_name, password_hash, role, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, email, display_name, password_hash, role, status, now, now),
            )
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            return _row(connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            return _row(connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone())

    def count_admins(self) -> int:
        with self.connection() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin' AND status = 'approved'"
                ).fetchone()[0]
            )

    def create_session(self, user_id: str, raw_token: str, csrf_token: str, days: int) -> None:
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=days)
        with self.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.isoformat(),))
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, csrf_token, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (token_hash(raw_token), user_id, csrf_token, now.isoformat(), expires.isoformat()),
            )
            connection.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?",
                (now.isoformat(), now.isoformat(), user_id),
            )

    def get_identity(self, raw_token: str | None) -> dict[str, Any] | None:
        if not raw_token:
            return None
        now = utcnow()
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT u.*, s.csrf_token AS session_csrf, s.expires_at AS session_expires_at
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (token_hash(raw_token), now),
            ).fetchone()
        return _row(row)

    def delete_session(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        with self.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(raw_token),))

    def delete_user_sessions(self, user_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def list_users(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM users"
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at DESC"
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def review_user(self, user_id: str, actor_id: str, status: str, role: str) -> None:
        now = utcnow()
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE users
                SET status = ?, role = ?, reviewed_by = ?, reviewed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, role, actor_id, now, now, user_id),
            )
            if status != "approved":
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def audit(
        self,
        action: str,
        target_type: str,
        target_id: str | None,
        *,
        actor_user_id: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO audit_log
                    (actor_user_id, action, target_type, target_id, details_json, ip_address, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    actor_user_id,
                    action,
                    target_type,
                    target_id,
                    json.dumps(details or {}, separators=(",", ":")),
                    ip_address,
                    utcnow(),
                ),
            )

    def recent_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT a.*, u.display_name AS actor_name, u.email AS actor_email
                    FROM audit_log a LEFT JOIN users u ON u.id = a.actor_user_id
                    ORDER BY a.created_at DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            ]

    def create_job(
        self,
        *,
        job_id: str,
        user_id: str,
        name: str,
        preset: str,
        params: dict[str, Any],
        files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = utcnow()
        input_files = [file for file in files if file["kind"] == "input"]
        input_bytes = sum(int(file["size_bytes"]) for file in input_files)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO jobs
                    (id, user_id, name, preset, params_json, input_count, input_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    user_id,
                    name,
                    preset,
                    json.dumps(params, separators=(",", ":")),
                    len(input_files),
                    input_bytes,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO job_files
                    (job_id, kind, filename, stored_path, size_bytes, sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        file["kind"],
                        file["filename"],
                        file["stored_path"],
                        file["size_bytes"],
                        file["sha256"],
                        now,
                    )
                    for file in files
                ],
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            job = _row(
                connection.execute(
                    """
                    SELECT j.*, u.display_name AS owner_name, u.email AS owner_email
                    FROM jobs j JOIN users u ON u.id = j.user_id
                    WHERE j.id = ?
                    """,
                    (job_id,),
                ).fetchone()
            )
        if job:
            job["params"] = json.loads(job.pop("params_json"))
        return job

    def list_jobs(self, user_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        where = "WHERE j.user_id = ?" if user_id else ""
        params: tuple[Any, ...] = (user_id, limit) if user_id else (limit,)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT j.*, u.display_name AS owner_name, u.email AS owner_email
                FROM jobs j JOIN users u ON u.id = j.user_id
                {where}
                ORDER BY j.created_at DESC LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def job_files(self, job_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM job_files WHERE job_id = ? ORDER BY kind, id",
                    (job_id,),
                ).fetchall()
            ]

    def get_job_file(self, file_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            return _row(
                connection.execute(
                    """
                    SELECT f.*, j.user_id, j.status AS job_status
                    FROM job_files f JOIN jobs j ON j.id = f.job_id
                    WHERE f.id = ?
                    """,
                    (file_id,),
                ).fetchone()
            )

    def job_stats(self, user_id: str | None = None) -> dict[str, int]:
        where = "WHERE user_id = ?" if user_id else ""
        params: tuple[Any, ...] = (user_id,) if user_id else ()
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT status, COUNT(*) AS count FROM jobs {where} GROUP BY status",
                params,
            ).fetchall()
        stats = {status: 0 for status in ("queued", "running", "succeeded", "failed", "canceled")}
        stats.update({row["status"]: row["count"] for row in rows})
        stats["total"] = sum(stats.values())
        return stats

    def request_cancel(self, job_id: str) -> None:
        now = utcnow()
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET cancel_requested = 1,
                    status = CASE WHEN status = 'queued' THEN 'canceled' ELSE status END,
                    stage = CASE WHEN status = 'queued' THEN 'Canceled' ELSE stage END,
                    progress_message = CASE WHEN status = 'queued' THEN 'Canceled before execution' ELSE 'Cancellation requested' END,
                    finished_at = CASE WHEN status = 'queued' THEN ? ELSE finished_at END
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (now, job_id),
            )

    def claim_next_job(self) -> dict[str, Any] | None:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM jobs WHERE status = 'queued' AND cancel_requested = 0 ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            now = utcnow()
            changed = connection.execute(
                """
                UPDATE jobs SET status = 'running', stage = 'Starting',
                    progress_message = 'Preparing the pipeline process', started_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, row["id"]),
            ).rowcount
            connection.commit()
            return self.get_job(row["id"]) if changed else None
        finally:
            connection.close()

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "status", "stage", "progress_message", "output_bytes", "return_code",
            "error_message", "cancel_requested", "log_path", "finished_at",
        }
        clean = {key: value for key, value in fields.items() if key in allowed}
        if not clean:
            return
        assignments = ", ".join(f"{key} = ?" for key in clean)
        with self.connection() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*clean.values(), job_id),
            )

    def add_output_file(
        self, job_id: str, filename: str, stored_path: str, size_bytes: int, sha256: str
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO job_files
                    (job_id, kind, filename, stored_path, size_bytes, sha256, created_at)
                VALUES (?, 'output', ?, ?, ?, ?, ?)
                """,
                (job_id, filename, stored_path, size_bytes, sha256, utcnow()),
            )

    def is_cancel_requested(self, job_id: str) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(row and row[0])

    def set_service_state(self, key: str, value: str) -> None:
        now = utcnow()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO service_state (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def get_service_state(self, key: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            return _row(
                connection.execute(
                    "SELECT * FROM service_state WHERE key = ?", (key,)
                ).fetchone()
            )

    def recover_interrupted_jobs(self) -> int:
        now = utcnow()
        with self.connection() as connection:
            return connection.execute(
                """
                UPDATE jobs SET status = 'failed', stage = 'Interrupted',
                    progress_message = 'The worker stopped before this job completed',
                    error_message = 'GPU worker interrupted', finished_at = ?
                WHERE status = 'running'
                """,
                (now,),
            ).rowcount

    def consume_rate_limit(self, key_hash: str, limit: int, window_seconds: int) -> bool:
        now = time.time()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT attempts, window_started FROM rate_limits WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
            if row is None or now - float(row["window_started"]) >= window_seconds:
                connection.execute(
                    """
                    INSERT INTO rate_limits (key_hash, attempts, window_started)
                    VALUES (?, 1, ?)
                    ON CONFLICT(key_hash) DO UPDATE SET attempts = 1, window_started = excluded.window_started
                    """,
                    (key_hash, now),
                )
                return True
            if int(row["attempts"]) >= limit:
                return False
            connection.execute(
                "UPDATE rate_limits SET attempts = attempts + 1 WHERE key_hash = ?",
                (key_hash,),
            )
            return True

    def clear_rate_limit(self, key_hash: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM rate_limits WHERE key_hash = ?", (key_hash,))
