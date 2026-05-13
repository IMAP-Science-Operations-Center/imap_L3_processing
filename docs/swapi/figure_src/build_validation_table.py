#!/usr/bin/env python3
"""Generate the integrator-validation table in `docs/swapi/solar-wind-moments.md`."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from imap_l3_processing.constants import (
    EV_TO_KELVIN,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    calculate_integral,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams

from figure_utils import (
    REPO_ROOT,
    bulk_velocity_rtn_from_swapi_angles,
    load_swapi_response,
    peak_esa_voltage_for_proton_bulk_speed,
    run_parallel_map,
)

_worker_state: types.SimpleNamespace | None = None

_REFERENCE_INTEGRALS_PATH = (
    REPO_ROOT / "tests" / "test_data" / "swapi" / "reference_integrals.csv"
)
_DOC_PATH = REPO_ROOT / "docs" / "swapi" / "solar-wind-moments.md"
_TABLE_BEGIN = "<!-- BEGIN: validation_table"
_TABLE_END = "<!-- END: validation_table -->"

_BANDS = [
    (r"$\lt 0.1$", 0.0, 0.1),
    ("$0.1$ – $1$", 0.1, 1.0),
    ("$1$ – $10$", 1.0, 10.0),
    ("$10$ – $10^2$", 10.0, 100.0),
    ("$10^2$ – $10^3$", 100.0, 1000.0),
    ("$10^3$ – $10^4$", 1000.0, 1e4),
    ("$10^4$ – $10^5$", 1e4, 1e5),
    ("$\\geq 10^5$", 1e5, np.inf),
]


def _build_table(rel: np.ndarray, refs_v: np.ndarray) -> str:
    lines = [
        "| Reference (Hz)  |     N |  Median |     95% |     99% |     Max |",
        "|-----------------|-------|---------|---------|---------|---------|",
    ]
    for label, lo, hi in _BANDS:
        mask = (refs_v >= lo) & (refs_v < hi)
        n = int(mask.sum())
        if n == 0:
            lines.append(
                f"| {label:<15s} | {n:>5d} |       — |       — |       — |       — |"
            )
            continue
        band = rel[mask]
        med = np.median(band) * 100
        p95 = np.percentile(band, 95) * 100
        p99 = np.percentile(band, 99) * 100
        mx = band.max() * 100
        lines.append(
            f"| {label:<15s} | {n:>5d} | {med:>6.2f}% | {p95:>6.2f}% | {p99:>6.2f}% | {mx:>6.2f}% |"
        )
    return "\n".join(lines)


def _update_doc(table_md: str) -> None:
    text = _DOC_PATH.read_text()
    begin_idx = text.find(_TABLE_BEGIN)
    end_idx = text.find(_TABLE_END)
    if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
        raise RuntimeError(
            f"Could not find '{_TABLE_BEGIN}' / '{_TABLE_END}' markers in {_DOC_PATH}"
        )
    begin_line_end = text.find("\n", begin_idx) + 1
    new_text = text[:begin_line_end] + table_md + "\n" + text[end_idx:]
    _DOC_PATH.write_text(new_text)


def _initialize_worker_state(
    rows: list[tuple[float, float, float, float, float]],
    peak_voltages: np.ndarray,
    swapi_response,
) -> None:
    global _worker_state
    _worker_state = types.SimpleNamespace(
        rows=rows,
        peak_voltages=peak_voltages,
        swapi_response=swapi_response,
        rotation_matrix=np.eye(3),
    )


def _process_one(i: int) -> float:
    state = _worker_state
    bulk_speed, azimuth, elevation, density, temperature_ev = state.rows[i]
    sw = SolarWindParams(
        density=density,
        bulk_velocity_rtn=bulk_velocity_rtn_from_swapi_angles(
            bulk_speed, azimuth, elevation
        ),
        temperature=temperature_ev * EV_TO_KELVIN,
        mass=PROTON_MASS_KG,
    )
    response_grid = state.swapi_response.get_response_grid(
        state.peak_voltages[i], PROTON_MASS_PER_CHARGE_M_P_PER_E
    )
    return calculate_integral(sw, response_grid, state.rotation_matrix)[0]


def main():
    print("Loading calibration data...")
    swapi_response = load_swapi_response()
    df = pd.read_csv(_REFERENCE_INTEGRALS_PATH)
    references = df["integral"].to_numpy()
    rows = [
        (
            float(row.bulk_speed),
            float(row.bulk_azimuth),
            float(row.bulk_elevation),
            float(row.density),
            float(row.temperature_ev),
        )
        for row in df.itertuples(index=False)
    ]

    print(f"Warming passband cache for {len(df)} rows...")
    peak_voltages = np.array(
        [peak_esa_voltage_for_proton_bulk_speed(r[0]) for r in rows]
    )
    swapi_response.warm_cache(peak_voltages)
    for unique_voltage in np.unique(peak_voltages):
        swapi_response.get_response_grid(
            float(unique_voltage), PROTON_MASS_PER_CHARGE_M_P_PER_E
        )

    _initialize_worker_state(rows, peak_voltages, swapi_response)
    optimized = np.array(
        run_parallel_map(_process_one, len(df), desc="integrals", chunksize=64)
    )

    valid = references > 0
    refs_v = references[valid]
    rel = np.abs(optimized[valid] / refs_v - 1.0)
    # Cases where both integrators round to ~0 give meaningless ratios; clamp.
    rel = np.minimum(rel, 1.0)

    print(f"\nMax |ratio - 1|:    {rel.max():.2%}")
    print(f"Median |ratio - 1|: {np.median(rel):.2%}")
    print(f"95th pct:           {np.percentile(rel, 95):.2%}")
    print(f"99th pct:           {np.percentile(rel, 99):.2%}")

    table_md = _build_table(rel, refs_v)
    print()
    print(table_md)
    _update_doc(table_md)
    print(f"\nUpdated {_DOC_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
