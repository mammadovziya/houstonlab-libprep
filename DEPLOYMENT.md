# Production deployment

LibPrep Cloud is designed for one Linux GPU server. The web process handles
accounts and uploads; a separate worker owns the GPU and executes one persistent
job queue. Caddy terminates HTTPS.

## Server requirements

- Linux host with an NVIDIA GPU of compute capability 7.0 or newer.
- NVIDIA driver compatible with CUDA 12.6 (driver 560.28 or newer).
- Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit.
- A DNS record pointing the chosen domain at the server.
- A large local filesystem for uploads and SDF results. Do not place the data
  directory on ephemeral container storage.

## First deployment

1. Copy `.env.example` to `.env` and set the domain, a random secret, data path,
   and first administrator credentials.
2. Create the data directory and make UID 10001 its owner.
3. Confirm the container runtime can see the GPU with NVIDIA's CUDA container
   smoke test.
4. Start the stack with `docker compose up -d --build`.
5. Open the configured domain, sign in as the bootstrap administrator, and
   remove `LIBPREP_ADMIN_PASSWORD` from `.env` after the account exists.

For a dedicated domain, leave `LIBPREP_ROOT_PATH` blank. If an existing reverse
proxy mounts the application below a path such as `/tools/libprep`, set that
value as `LIBPREP_ROOT_PATH` and configure the upstream proxy to forward the
same prefix.

Caddy obtains and renews the TLS certificate automatically when ports 80 and
443 are reachable and DNS is correct.

## Operations

- `docker compose logs -f web worker` follows application and GPU worker logs.
- `docker compose restart worker` restarts only the queue worker. A run that was
  active during an unexpected worker restart is marked failed rather than
  duplicated.
- Back up the entire configured data path. It contains the SQLite database,
  uploaded catalogues, manifests, logs, and result files.
- Upgrade with a fresh backup, `git pull`, then `docker compose up -d --build`.

SQLite is configured in WAL mode and is appropriate for this single-server,
single-GPU queue. Move account and queue state to a managed database before
running multiple application servers or workers across hosts.

## Path-based deployment

To mount LibPrep below an existing HTTPS site, run it on a loopback port and
forward a path such as `/tools/libprep` from the site's reverse proxy.

Set these values in the LibPrep service environment:

```bash
LIBPREP_ENV=production
LIBPREP_ROOT_PATH=/tools/libprep
LIBPREP_ALLOWED_HOSTS=lab.example.org
LIBPREP_SECURE_COOKIES=true
LIBPREP_FORCE_HTTPS=false
```

Run the web process on loopback so it is not directly exposed:

```bash
uvicorn webapp.app:app \
  --host 127.0.0.1 \
  --port 9001 \
  --proxy-headers \
  --forwarded-allow-ips=127.0.0.1
```

The equivalent Nginx configuration is:

```nginx
location = /tools/libprep {
    return 301 /tools/libprep/;
}

location /tools/libprep/ {
    client_max_body_size 10G;
    proxy_read_timeout 86400;
    proxy_pass http://127.0.0.1:9001/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Prefix /tools/libprep;
}
```

The trailing slash on `proxy_pass` is required: it strips `/tools/libprep/`
before forwarding to FastAPI, while `LIBPREP_ROOT_PATH` keeps generated links,
static assets, cookies, and redirects under the public prefix.
