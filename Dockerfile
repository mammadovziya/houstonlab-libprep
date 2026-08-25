FROM condaforge/miniforge3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LIBPREP_DATA_DIR=/data \
    LIBPREP_PIPELINE_PYTHON=/opt/conda/bin/python \
    TMPDIR=/data/tmp

WORKDIR /app

COPY environment.yml /tmp/environment.yml
RUN mamba env update --name base --file /tmp/environment.yml && \
    mamba clean --all --yes

COPY requirements-web.txt /tmp/requirements-web.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-web.txt

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin libprep && \
    mkdir -p /data && chown -R libprep:libprep /data /app

COPY --chown=libprep:libprep . /app

USER libprep
EXPOSE 8000

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
