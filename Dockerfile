# ---------------------------------------------------------------------------
# Builder stage: resolve and install the locked environment with uv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# uv provides fast, reproducible, lockfile-driven installs.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install third-party dependencies first so this layer is cached and only
# re-runs when the lockfile or project metadata changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Install the project itself against the already-built dependency layer.
COPY imap_l3_processing ./imap_l3_processing
COPY imap_l3_data_processor.py ./
RUN uv sync --locked --no-dev

# ---------------------------------------------------------------------------
# Runtime stage: copy just the venv and app code onto a clean base image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgfortran5 is required by the Fortran GLOWS L3e survProb* executables.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Run everything through the prebuilt virtual environment.
ENV VIRTUAL_ENV="/app/.venv" PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY imap_l3_processing ./imap_l3_processing
COPY imap_l3_data_processor.py .

# The GLOWS L3e survival-probability executables are invoked as ./survProb*
# from the current working directory, so expose them at the WORKDIR root.
RUN cp imap_l3_processing/glows/l3e/l3e_toolkit/survProb* .

# Make sure executables have appropriate permissions
RUN chmod +x survProbLo survProbHi survProbUltra

# Working directory expected at runtime.
RUN mkdir -p temp_cdf_data

ENTRYPOINT ["python", "imap_l3_data_processor.py"]
