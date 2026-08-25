from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from pathlib import Path

from pwdlib import PasswordHash


password_hasher = PasswordHash.recommended()
_DUMMY_PASSWORD_HASH = password_hasher.hash("not-a-real-user-password")
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def normalise_email(value: str) -> str:
    return value.strip().lower()


def valid_email(value: str) -> bool:
    return len(value) <= 254 and bool(EMAIL_RE.fullmatch(value))


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, encoded: str | None) -> bool:
    candidate = encoded or _DUMMY_PASSWORD_HASH
    try:
        valid = password_hasher.verify(password, candidate)
    except Exception:
        valid = False
    return bool(encoded) and valid


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def public_csrf_token(seed: str, secret_key: str) -> str:
    return hmac.new(
        secret_key.encode("utf-8"), seed.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def verify_public_csrf(seed: str | None, supplied: str, secret_key: str) -> bool:
    if not seed or not supplied:
        return False
    return hmac.compare_digest(public_csrf_token(seed, secret_key), supplied)


def safe_upload_name(filename: str | None, fallback: str = "upload.smi") -> str:
    name = Path(filename or fallback).name.strip().replace("\x00", "")
    name = SAFE_NAME_RE.sub("_", name).strip("._")
    return (name or fallback)[:180]
