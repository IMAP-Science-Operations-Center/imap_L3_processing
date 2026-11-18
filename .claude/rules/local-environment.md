# Local Environment

This repo is developed on both macOS and Windows. Before running commands, determine the current OS from the environment information you are given (OS version, shell, workspace path) and follow the matching section below. Never assume one OS from the other's conventions.

## Shared across all machines

- Run the processor and tests through `uv` (e.g. `uv run python -m unittest discover tests`, `uv run python imap_l3_data_processor.py ...`).
- Docker Desktop must be running for the Docker-based scripts (e.g. `run_local_using_docker.sh`).
- CI (`.github/workflows/test.yml`) and the Docker image run on Linux only, so behavior that differs on macOS or Windows (paths, line endings, case sensitivity) will not be caught by CI.

## macOS (Darwin, zsh)

- Use Unix/POSIX shell commands (`ls`, `cp`, `mv`, `mkdir -p`, `export VAR=value`).
- Use forward slashes in paths; the virtual environment's executables are in `.venv/bin/`.
- Run the repo's `.sh` scripts directly from zsh/bash.
- For deletions, prefer `trash` over `rm`.
- Homebrew (`brew install ...`) is available for system packages (e.g. `hdf5`, `netcdf`, `git-lfs`).

## Windows (PowerShell)

- Use PowerShell syntax (`$env:VAR = "value"`, `Remove-Item`, `New-Item -ItemType Directory`).
- The virtual environment's executables are in `.venv\Scripts\` rather than `.venv/bin/`.
- Append `exit $LASTEXITCODE` after `uv run` commands in scripted contexts so failures propagate.
- Use backslashes in native paths where required.
- Run the repo's `.sh` scripts (e.g. `run_local_using_docker.sh`) from Git Bash, not PowerShell.
- For deletions, prefer moving to the Recycle Bin over permanent deletion.
