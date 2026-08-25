from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from webapp.config import Settings
from webapp.db import Database
from webapp.security import (
    hash_password,
    new_csrf_token,
    new_session_token,
    normalise_email,
    public_csrf_token,
    safe_upload_name,
    valid_email,
    verify_password,
    verify_public_csrf,
)


BASE_DIR = Path(__file__).resolve().parent
SESSION_COOKIE = "libprep_session"
PUBLIC_CSRF_COOKIE = "libprep_form"
INPUT_EXTENSIONS = {".smi", ".smiles", ".cxsmiles", ".txt", ".tsv"}
SMARTS_EXTENSIONS = {".smarts", ".txt"}


def human_bytes(value: int | None) -> str:
    size = float(value or 0)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return "0 B"


def display_time(value: str | None) -> str:
    if not value:
        return "Not available"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone().strftime("%d %b %Y · %H:%M")
    except ValueError:
        return value


def client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()[:64]
    return request.client.host[:64] if request.client else None


def is_staff(user: dict[str, Any] | None) -> bool:
    return bool(user and user["role"] in {"admin", "moderator"})


def can_access_job(user: dict[str, Any], job: dict[str, Any]) -> bool:
    return is_staff(user) or job["user_id"] == user["id"]


def validate_job_params(form: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        preset = str(form.get("preset", "docking"))
        if preset not in {"docking", "enumerate"}:
            return None, "Choose a valid preparation preset."
        params = {
            "n_conformers": int(form.get("n_conformers", 1)),
            "max_unspecified_stereo": int(form.get("max_unspecified_stereo", 2)),
            "max_tautomers": int(form.get("max_tautomers", 5)),
            "ph_min": float(form.get("ph_min", 7.4)),
            "ph_max": float(form.get("ph_max", 7.4)),
            "tautomers": form.get("tautomers") == "1",
            "ionise": form.get("ionise") == "1",
            "conformers": form.get("conformers") == "1",
            "chunk_size": int(form.get("chunk_size", 100000)),
            "batch_size": int(form.get("batch_size", 500)),
            "batches_per_gpu": int(form.get("batches_per_gpu", 4)),
            "preprocessing_threads": int(form.get("preprocessing_threads", 8)),
            "mmff_max_iters": int(form.get("mmff_max_iters", 200)),
            "pains_backend": str(form.get("pains_backend", "auto")),
            "save_intermediates": form.get("save_intermediates") == "1",
        }
    except (TypeError, ValueError):
        return None, "One or more pipeline settings are not valid numbers."
    limits = {
        "n_conformers": (1, 50),
        "max_unspecified_stereo": (0, 4),
        "max_tautomers": (1, 20),
        "chunk_size": (1_000, 1_000_000),
        "batch_size": (1, 10_000),
        "batches_per_gpu": (1, 32),
        "preprocessing_threads": (1, 128),
        "mmff_max_iters": (1, 5_000),
    }
    for key, (minimum, maximum) in limits.items():
        if not minimum <= params[key] <= maximum:
            label = key.replace("_", " ")
            return None, f"{label.capitalize()} must be between {minimum:,} and {maximum:,}."
    if not (0 <= params["ph_min"] <= params["ph_max"] <= 14):
        return None, "The pH range must be between 0 and 14, with minimum no greater than maximum."
    if params["pains_backend"] not in {"auto", "cpu", "gpu"}:
        return None, "Choose a valid PAINS filtering backend."
    params["preset"] = preset
    return params, None


async def persist_upload(upload: UploadFile, destination: Path, max_bytes: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("File exceeds the configured upload limit")
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("Uploaded files cannot be empty")
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    database = Database(settings.database_path)
    root_path = settings.root_path
    development_admin_username = os.getenv("LIBPREP_ADMIN_USERNAME", "admin").strip().lower()
    development_admin_email = normalise_email(
        os.getenv("LIBPREP_ADMIN_EMAIL", "")
        or ("admin@libprep.local" if settings.environment != "production" else "")
    )
    development_admin_password = os.getenv("LIBPREP_ADMIN_PASSWORD", "") or (
        "admin" if settings.environment != "production" else ""
    )

    def app_url(path: str = "/") -> str:
        path = path if path.startswith("/") else f"/{path}"
        return f"{root_path}{path}" or "/"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        email = development_admin_email
        password = development_admin_password
        name = os.getenv("LIBPREP_ADMIN_NAME", "Platform administrator").strip()
        if email and password and not database.get_user_by_email(email):
            if not valid_email(email):
                raise RuntimeError("Bootstrap administrator credentials are invalid")
            if settings.environment == "production" and len(password) < 12:
                raise RuntimeError("Bootstrap administrator password must be at least 12 characters")
            user = database.create_user(
                email=email,
                display_name=name[:80],
                password_hash=hash_password(password),
                role="admin",
                status="approved",
            )
            database.audit("user.bootstrap_admin", "user", user["id"], actor_user_id=user["id"])
        yield

    app = FastAPI(
        title="LibPrep",
        description="GPU-accelerated chemical library preparation",
        version="1.0.0",
        lifespan=lifespan,
        root_path=root_path,
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.database = database
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    templates.env.filters["filesize"] = human_bytes
    templates.env.filters["displaytime"] = display_time
    templates.env.globals["app_url"] = app_url
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    if settings.force_https:
        app.add_middleware(HTTPSRedirectMiddleware)

    @app.middleware("http")
    async def identity_and_security_headers(request: Request, call_next):
        request.state.identity = database.get_identity(request.cookies.get(SESSION_COOKIE))
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if settings.secure_cookies:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
            "font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )
        if request.url.path.startswith(("/dashboard", "/jobs", "/admin", "/login", "/register")):
            response.headers["Cache-Control"] = "no-store"
        return response

    def identity(request: Request) -> dict[str, Any] | None:
        return request.state.identity

    def require_user(request: Request) -> dict[str, Any]:
        user = identity(request)
        if not user:
            raise HTTPException(status_code=303, headers={"Location": app_url("/login")})
        if user["status"] != "approved":
            raise HTTPException(status_code=303, headers={"Location": app_url("/pending")})
        return user

    def require_staff(request: Request) -> dict[str, Any]:
        user = require_user(request)
        if not is_staff(user):
            raise HTTPException(status_code=403, detail="Staff access required")
        return user

    def render(
        request: Request,
        name: str,
        context: dict[str, Any] | None = None,
        *,
        status_code: int = 200,
        public_form: bool = False,
    ):
        user = identity(request)
        values: dict[str, Any] = {
            "title": "LibPrep",
            "current_user": user,
            "is_staff": is_staff(user),
            "registration_enabled": settings.registration_enabled,
            "app_root": root_path,
        }
        values.update(context or {})
        if user:
            values["csrf_token"] = user["session_csrf"]
        elif public_form:
            seed = request.cookies.get(PUBLIC_CSRF_COOKIE) or secrets.token_urlsafe(24)
            values["csrf_token"] = public_csrf_token(seed, settings.secret_key)
        response = templates.TemplateResponse(
            request=request,
            name=name,
            context=values,
            status_code=status_code,
        )
        if public_form and not user and not request.cookies.get(PUBLIC_CSRF_COOKIE):
            response.set_cookie(
                PUBLIC_CSRF_COOKIE,
                seed,
                path=app_url("/"),
                max_age=3600,
                httponly=True,
                secure=settings.secure_cookies,
                samesite="lax",
            )
        return response

    def csrf_ok(request: Request, supplied: str) -> bool:
        user = identity(request)
        if user:
            return secrets.compare_digest(user["session_csrf"], supplied or "")
        return verify_public_csrf(
            request.cookies.get(PUBLIC_CSRF_COOKIE), supplied, settings.secret_key
        )

    @app.exception_handler(403)
    async def forbidden(request: Request, _: HTTPException):
        return render(
            request,
            "error.html",
            {"heading": "Access denied", "message": "You do not have permission to view this page."},
            status_code=403,
        )

    @app.get("/", include_in_schema=False)
    async def landing(request: Request):
        return render(request, "landing.html")

    @app.get("/healthz", include_in_schema=False)
    async def healthcheck():
        try:
            database.get_service_state("worker")
        except sqlite3.Error:
            return JSONResponse({"status": "error"}, status_code=503)
        return {"status": "ok"}

    @app.get("/register", include_in_schema=False)
    async def register_page(request: Request):
        if identity(request):
            return RedirectResponse(app_url("/dashboard"), status_code=303)
        if not settings.registration_enabled:
            return render(
                request,
                "error.html",
                {"heading": "Registration is closed", "message": "Ask a platform administrator to create an account."},
                status_code=403,
            )
        return render(request, "register.html", public_form=True)

    @app.post("/register", include_in_schema=False)
    async def register_submit(request: Request):
        if not settings.registration_enabled:
            raise HTTPException(status_code=403)
        form = await request.form()
        if not csrf_ok(request, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403)
        registration_key = hashlib.sha256(
            f"register:{client_ip(request) or 'unknown'}".encode("utf-8")
        ).hexdigest()
        if not database.consume_rate_limit(registration_key, limit=5, window_seconds=3600):
            return render(
                request,
                "register.html",
                {"error": "Too many registration attempts. Try again later."},
                status_code=429,
                public_form=True,
            )
        email = normalise_email(str(form.get("email", "")))
        name = " ".join(str(form.get("display_name", "")).strip().split())
        password = str(form.get("password", ""))
        confirmation = str(form.get("password_confirmation", ""))
        error = None
        if not 2 <= len(name) <= 80:
            error = "Enter your full name (2 to 80 characters)."
        elif not valid_email(email):
            error = "Enter a valid email address."
        elif not 12 <= len(password) <= 128:
            error = "Use a password between 12 and 128 characters."
        elif password != confirmation:
            error = "The password confirmation does not match."
        elif database.get_user_by_email(email):
            error = "An account already exists for this email address."
        if error:
            return render(
                request,
                "register.html",
                {"error": error, "form_email": email, "form_name": name},
                status_code=422,
                public_form=True,
            )
        try:
            user = database.create_user(
                email=email,
                display_name=name,
                password_hash=hash_password(password),
            )
        except sqlite3.IntegrityError:
            return render(
                request,
                "register.html",
                {"error": "An account already exists for this email address."},
                status_code=422,
                public_form=True,
            )
        database.audit(
            "user.registered", "user", user["id"], details={"email": email}, ip_address=client_ip(request)
        )
        return RedirectResponse(app_url("/login?registered=1"), status_code=303)

    @app.get("/login", include_in_schema=False)
    async def login_page(request: Request):
        if identity(request) and identity(request)["status"] == "approved":
            return RedirectResponse(app_url("/dashboard"), status_code=303)
        return render(
            request,
            "login.html",
            {"registered": request.query_params.get("registered") == "1"},
            public_form=True,
        )

    @app.post("/login", include_in_schema=False)
    async def login_submit(request: Request):
        form = await request.form()
        if not csrf_ok(request, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403)
        supplied_identity = normalise_email(
            str(form.get("identity", "") or form.get("email", ""))
        )
        email = (
            development_admin_email
            if development_admin_email and supplied_identity == development_admin_username
            else supplied_identity
        )
        password = str(form.get("password", ""))
        login_key = hashlib.sha256(
            f"login:{email}:{client_ip(request) or 'unknown'}".encode("utf-8")
        ).hexdigest()
        if not database.consume_rate_limit(login_key, limit=8, window_seconds=900):
            return render(
                request,
                "login.html",
                {
                    "error": "Too many sign-in attempts. Try again in 15 minutes.",
                    "form_identity": supplied_identity,
                },
                status_code=429,
                public_form=True,
            )
        user = database.get_user_by_email(email)
        if not verify_password(password, user["password_hash"] if user else None):
            database.audit(
                "auth.login_failed", "user", user["id"] if user else None,
                details={"email": email}, ip_address=client_ip(request),
            )
            return render(
                request,
                "login.html",
                {
                    "error": "Username/email or password is incorrect.",
                    "form_identity": supplied_identity,
                },
                status_code=401,
                public_form=True,
            )
        if user["status"] != "approved":
            return render(
                request,
                "login.html",
                {"account_status": user["status"], "form_identity": supplied_identity},
                status_code=403,
                public_form=True,
            )
        raw_token = new_session_token()
        csrf_token = new_csrf_token()
        database.clear_rate_limit(login_key)
        database.create_session(user["id"], raw_token, csrf_token, settings.session_days)
        database.audit("auth.login", "user", user["id"], actor_user_id=user["id"], ip_address=client_ip(request))
        response = RedirectResponse(app_url("/dashboard"), status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            raw_token,
            path=app_url("/"),
            max_age=settings.session_days * 86400,
            httponly=True,
            secure=settings.secure_cookies,
            samesite="lax",
        )
        response.delete_cookie(PUBLIC_CSRF_COOKIE, path=app_url("/"))
        return response

    @app.get("/pending", include_in_schema=False)
    async def pending_page(request: Request):
        user = identity(request)
        if not user:
            return RedirectResponse(app_url("/login"), status_code=303)
        if user["status"] == "approved":
            return RedirectResponse(app_url("/dashboard"), status_code=303)
        return render(request, "pending.html")

    @app.post("/logout", include_in_schema=False)
    async def logout(request: Request):
        form = await request.form()
        if not csrf_ok(request, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403)
        user = identity(request)
        database.delete_session(request.cookies.get(SESSION_COOKIE))
        if user:
            database.audit("auth.logout", "user", user["id"], actor_user_id=user["id"])
        response = RedirectResponse(app_url("/"), status_code=303)
        response.delete_cookie(SESSION_COOKIE, path=app_url("/"))
        return response

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard(request: Request):
        user = require_user(request)
        staff = is_staff(user)
        jobs = database.list_jobs(None if staff else user["id"])
        stats = database.job_stats(None if staff else user["id"])
        worker = database.get_service_state("worker")
        worker_online = False
        if worker:
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(worker["updated_at"])
                worker_online = age.total_seconds() < 30
            except ValueError:
                pass
        pending_count = len(database.list_users("pending")) if staff else 0
        return render(
            request,
            "dashboard.html",
            {"jobs": jobs, "stats": stats, "worker_online": worker_online, "pending_count": pending_count},
        )

    @app.get("/jobs/new", include_in_schema=False)
    async def new_job_page(request: Request):
        require_user(request)
        return render(request, "job_new.html")

    @app.post("/jobs", include_in_schema=False)
    async def create_job_route(request: Request):
        user = require_user(request)
        form = await request.form()
        if not csrf_ok(request, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403)
        params, error = validate_job_params(form)
        uploads = [item for item in form.getlist("files") if getattr(item, "filename", "")]
        custom_smarts = form.get("custom_smarts")
        if not uploads:
            error = error or "Upload at least one SMILES or CXSMILES file."
        if len(uploads) > 20:
            error = error or "A job can contain at most 20 supplier files."
        for upload in uploads:
            if Path(safe_upload_name(upload.filename)).suffix.lower() not in INPUT_EXTENSIONS:
                error = error or f"Unsupported input file type: {upload.filename}"
                break
        if getattr(custom_smarts, "filename", ""):
            if Path(safe_upload_name(custom_smarts.filename)).suffix.lower() not in SMARTS_EXTENSIONS:
                error = error or "Custom SMARTS must be a .smarts or .txt file."
        if error:
            return render(request, "job_new.html", {"error": error}, status_code=422)

        job_id = uuid.uuid4().hex[:16]
        job_dir = settings.data_dir / "jobs" / job_id
        input_dir = job_dir / "input"
        stored: list[dict[str, Any]] = []
        total_bytes = 0
        try:
            for index, upload in enumerate(uploads, 1):
                original = safe_upload_name(upload.filename)
                stored_name = f"{index:02d}_{original}"
                details = await persist_upload(upload, input_dir / stored_name, settings.max_upload_bytes)
                total_bytes += details["size_bytes"]
                if total_bytes > settings.max_upload_bytes:
                    raise ValueError("Combined uploads exceed the configured job limit")
                stored.append({
                    "kind": "input", "filename": original,
                    "stored_path": str((input_dir / stored_name).resolve()), **details,
                })
            if getattr(custom_smarts, "filename", ""):
                original = safe_upload_name(custom_smarts.filename, "custom.smarts")
                path = input_dir / f"custom_{original}"
                details = await persist_upload(custom_smarts, path, min(settings.max_upload_bytes, 10 * 1024**2))
                stored.append({
                    "kind": "custom_smarts", "filename": original,
                    "stored_path": str(path.resolve()), **details,
                })
        except (OSError, ValueError) as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            return render(request, "job_new.html", {"error": str(exc)}, status_code=422)

        name = " ".join(str(form.get("name", "")).strip().split())[:100]
        if not name:
            name = Path(stored[0]["filename"]).stem[:100] or f"Library {job_id[:6]}"
        database.create_job(
            job_id=job_id, user_id=user["id"], name=name,
            preset=params.pop("preset"), params=params, files=stored,
        )
        database.audit(
            "job.created", "job", job_id, actor_user_id=user["id"],
            details={"input_count": len(uploads), "input_bytes": total_bytes},
            ip_address=client_ip(request),
        )
        return RedirectResponse(app_url(f"/jobs/{job_id}"), status_code=303)

    @app.get("/jobs/{job_id}", include_in_schema=False)
    async def job_detail(request: Request, job_id: str):
        user = require_user(request)
        job = database.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404)
        if not can_access_job(user, job):
            raise HTTPException(status_code=403)
        files = database.job_files(job_id)
        log_tail = ""
        if job.get("log_path"):
            path = Path(job["log_path"])
            if path.is_file():
                try:
                    log_tail = "".join(path.read_text(errors="replace").splitlines(keepends=True)[-80:])
                except OSError:
                    pass
        return render(request, "job_detail.html", {"job": job, "files": files, "log_tail": log_tail})

    @app.get("/api/jobs/{job_id}", include_in_schema=False)
    async def job_status(request: Request, job_id: str):
        user = require_user(request)
        job = database.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404)
        if not can_access_job(user, job):
            raise HTTPException(status_code=403)
        return {
            "id": job["id"], "status": job["status"], "stage": job["stage"],
            "message": job["progress_message"], "output_bytes": job["output_bytes"],
            "finished_at": job["finished_at"],
        }

    @app.post("/jobs/{job_id}/cancel", include_in_schema=False)
    async def cancel_job(request: Request, job_id: str):
        user = require_user(request)
        form = await request.form()
        if not csrf_ok(request, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403)
        job = database.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404)
        if not can_access_job(user, job):
            raise HTTPException(status_code=403)
        database.request_cancel(job_id)
        database.audit("job.cancel_requested", "job", job_id, actor_user_id=user["id"])
        return RedirectResponse(app_url(f"/jobs/{job_id}"), status_code=303)

    @app.get("/files/{file_id}/download", include_in_schema=False)
    async def download_file(request: Request, file_id: int):
        user = require_user(request)
        record = database.get_job_file(file_id)
        if not record:
            raise HTTPException(status_code=404)
        if not is_staff(user) and record["user_id"] != user["id"]:
            raise HTTPException(status_code=403)
        path = Path(record["stored_path"]).resolve()
        job_root = (settings.data_dir / "jobs" / record["job_id"]).resolve()
        if not path.is_relative_to(job_root) or not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, filename=record["filename"], media_type="application/octet-stream")

    @app.get("/admin/users", include_in_schema=False)
    async def admin_users(request: Request):
        require_staff(request)
        status = request.query_params.get("status")
        if status not in {None, "pending", "approved", "rejected", "disabled"}:
            status = None
        return render(request, "admin_users.html", {"users": database.list_users(status), "status_filter": status})

    @app.post("/admin/users/{user_id}/review", include_in_schema=False)
    async def review_user(request: Request, user_id: str):
        actor = require_staff(request)
        form = await request.form()
        if not csrf_ok(request, str(form.get("csrf_token", ""))):
            raise HTTPException(status_code=403)
        target = database.get_user(user_id)
        if not target:
            raise HTTPException(status_code=404)
        status = str(form.get("status", ""))
        role = str(form.get("role", "user"))
        if status not in {"approved", "rejected", "disabled"} or role not in {"admin", "moderator", "user"}:
            raise HTTPException(status_code=422)
        if actor["role"] != "admin":
            if target["role"] != "user":
                raise HTTPException(status_code=403)
            role = "user"
        if target["id"] == actor["id"] and status != "approved":
            return RedirectResponse(app_url("/admin/users?error=self"), status_code=303)
        if (
            target["role"] == "admin"
            and (status != "approved" or role != "admin")
            and database.count_admins() <= 1
        ):
            return RedirectResponse(app_url("/admin/users?error=last-admin"), status_code=303)
        database.review_user(user_id, actor["id"], status, role)
        database.audit(
            "user.reviewed", "user", user_id, actor_user_id=actor["id"],
            details={"status": status, "role": role}, ip_address=client_ip(request),
        )
        return RedirectResponse(app_url("/admin/users"), status_code=303)

    @app.get("/admin/audit", include_in_schema=False)
    async def audit_log(request: Request):
        require_staff(request)
        return render(request, "audit_log.html", {"events": database.recent_audit()})

    return app


app = create_app()
