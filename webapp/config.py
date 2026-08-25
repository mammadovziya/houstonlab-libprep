from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    root_path: str = field(
        default_factory=lambda: os.getenv("LIBPREP_ROOT_PATH", "").strip()
    )
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("LIBPREP_DATA_DIR", PROJECT_DIR / "data")
        ).expanduser().resolve()
    )
    database_path: Path | None = None
    secret_key: str = field(
        default_factory=lambda: os.getenv("LIBPREP_SECRET_KEY") or secrets.token_urlsafe(48)
    )
    environment: str = field(default_factory=lambda: os.getenv("LIBPREP_ENV", "development"))
    secure_cookies: bool = field(
        default_factory=lambda: _as_bool(
            os.getenv("LIBPREP_SECURE_COOKIES"),
            os.getenv("LIBPREP_ENV", "development") == "production",
        )
    )
    force_https: bool = field(
        default_factory=lambda: _as_bool(os.getenv("LIBPREP_FORCE_HTTPS"), False)
    )
    registration_enabled: bool = field(
        default_factory=lambda: _as_bool(os.getenv("LIBPREP_REGISTRATION_ENABLED"), True)
    )
    allowed_hosts: list[str] = field(
        default_factory=lambda: [
            host.strip()
            for host in os.getenv(
                "LIBPREP_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver"
            ).split(",")
            if host.strip()
        ]
    )
    max_upload_bytes: int = field(
        default_factory=lambda: int(
            float(os.getenv("LIBPREP_MAX_UPLOAD_GB", "10")) * 1024**3
        )
    )
    session_days: int = field(
        default_factory=lambda: int(os.getenv("LIBPREP_SESSION_DAYS", "7"))
    )
    pipeline_python: str = field(
        default_factory=lambda: os.getenv("LIBPREP_PIPELINE_PYTHON", "python")
    )

    def __post_init__(self) -> None:
        if self.root_path and not self.root_path.startswith("/"):
            self.root_path = f"/{self.root_path}"
        self.root_path = self.root_path.rstrip("/")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "tmp").mkdir(parents=True, exist_ok=True)
        if self.database_path is None:
            self.database_path = self.data_dir / "libprep.sqlite3"
        else:
            self.database_path = Path(self.database_path).expanduser().resolve()
        if self.environment == "production" and not os.getenv("LIBPREP_SECRET_KEY"):
            raise RuntimeError("LIBPREP_SECRET_KEY is required in production")
