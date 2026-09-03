#!/usr/bin/env python3
"""
Plot one flight chunk's mean He+ PUI spectrum: observed coincidence rate vs the
production forward model and the xarray + pint reference forward model, with
the PUI fit energy window shaded.

Runs the production helium PUI fit over the whole day via
`scripts/swapi/fit_and_plot_pui.py`, then renders the single chunk nearest
15:54:05 UT via `scripts/swapi/view_one_pui_spectrum.py`.

Requires the environment variable IMAP_API_KEY to be set (both underlying
scripts download their L2 and SPICE inputs from the SDC).

TODO: refactor this into a self-contained figure_src script, the way
plot_alpha_peak_finding.py resolves and downloads its own inputs, instead of
driving two scripts/swapi/ entry points over a /tmp pickle handoff. Three
things have to be sorted out first:
  1. scripts/swapi/pui_xarray_reference_50sweep.py imports pint and
     pint_xarray, neither of which is in pyproject.toml, so this figure cannot
     be regenerated from the lockfile alone.
  2. view_one_pui_spectrum.py reads the whole-day fit and spectrogram pickles
     that fit_and_plot_pui.py leaves in /tmp; only the one selected chunk is
     actually needed, so the fit could be run for that chunk alone.
  3. Both scripts do all their work at module level under argparse, so neither
     can be imported and called directly.

Output: docs/swapi/figures/pui_flight_xarray_comparison.svg
Usage:  uv run python docs/swapi/figure_src/plot_pui_flight_xarray_comparison.py
        uv run python docs/swapi/figure_src/plot_pui_flight_xarray_comparison.py --use-cache
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from figure_utils import FIGURES_DIR

_DATE = "2026-01-01"
_CHUNK_TIME = "15:54:05"
_OUTPUT_PATH = FIGURES_DIR / "pui_flight_xarray_comparison.svg"

_SCRIPTS_DIR = REPO_ROOT / "scripts" / "swapi"
_FIT_SCRIPT = _SCRIPTS_DIR / "fit_and_plot_pui.py"
_VIEW_SCRIPT = _SCRIPTS_DIR / "view_one_pui_spectrum.py"


def _run(command: list[str]) -> None:
    # Claude: both scripts import from the `scripts.swapi` package, which is not installed into the venv, so the repo root has to be on the child's path.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment)
    if result.returncode != 0:
        sys.exit(f"{Path(command[1]).name} failed with exit code {result.returncode}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument(
        "--use-cache", action="store_true",
        help="Reuse the /tmp fit pickles and local input cache from a prior run "
             "instead of refitting the day and re-querying the SDC.")
    arguments = argument_parser.parse_args()

    if not arguments.use_cache and "IMAP_API_KEY" not in os.environ:
        sys.exit("IMAP_API_KEY environment variable is required.")

    # Claude: fit_and_plot_pui.py calls plt.show() and blocks unless --output-dir is set; its own PNG is a byproduct we do not keep.
    with tempfile.TemporaryDirectory() as scratch_dir:
        fit_command = [
            sys.executable, str(_FIT_SCRIPT), _DATE,
            "--output-dir", scratch_dir,
        ]
        if arguments.use_cache:
            fit_command.append("--use-cache")
        _run(fit_command)

    view_command = [
        sys.executable, str(_VIEW_SCRIPT), _DATE, _CHUNK_TIME,
        "--output-path", str(_OUTPUT_PATH),
    ]
    if arguments.use_cache:
        view_command.append("--offline")
    _run(view_command)

    print(f"Wrote {_OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
