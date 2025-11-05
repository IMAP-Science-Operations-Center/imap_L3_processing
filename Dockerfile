FROM python:3.12-slim
RUN apt-get update && apt-get install -y git libgfortran5
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
RUN mkdir temp_cdf_data
RUN mkdir -p /mnt/spice
COPY imap_l3_processing/glows/l3e/l3e_toolkit/survProbHi survProbHi
COPY imap_l3_processing/glows/l3e/l3e_toolkit/survProbLo survProbLo
COPY imap_l3_processing/glows/l3e/l3e_toolkit/survProbUltra survProbUltra
RUN chmod +x survProbHi
RUN chmod +x survProbLo
RUN chmod +x survProbUltra
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --locked --no-install-project
COPY imap_l3_processing imap_l3_processing
COPY imap_l3_data_processor.py .
RUN uv sync --locked
ENTRYPOINT ["uv", "run", "imap_l3_data_processor.py"]