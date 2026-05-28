#!/usr/bin/env python3
"""
Illustrate the shape of the Vasyliunas-Siscoe filled-shell PUI distribution
f_PUI(v') in the solar-wind comoving frame.

For a fixed cutoff speed v_b and ionization rate, the distribution is plotted
against the comoving speed v' for several cooling indices alpha_PUI. Two
features the moment integrals depend on are visible:

  * the hard cutoff at v' = v_b (w = 1), above which f_PUI is identically zero;
  * the competition between the w^(alpha - 3) factor (which grows toward small
    w) and the neutral-helium density n(r w^alpha) (which is depleted close to
    the Sun, i.e. small w), producing a peaked shell rather than a divergence.

Output: docs/swapi/figures/pui_distribution.svg
Usage:  uv run python docs/swapi/figure_src/plot_pui_distribution.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_utils import FIGURES_DIR

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)

_CUTOFF_SPEED_KMS = 450.0
_SW_SPEED_INERTIAL_KMS = 450.0
_IONIZATION_RATE_HZ = 2e-7
_INFLOW_PSI_DEG = 75.0
_COOLING_INDICES = [1.5, 2.5, 3.5]

_DENSITY_LUT_PATH = (
    REPO_ROOT / "instrument_team_data" / "swapi" / "density-of-neutral-helium-lut.dat"
)


def main():
    distribution = VasyliunasSiscoeDistribution(
        ephemeris_time=0.0,
        solar_wind_speed_inertial_frame=_SW_SPEED_INERTIAL_KMS,
        density_of_neutral_helium_lookup_table=DensityOfNeutralHeliumLookupTable.from_file(
            _DENSITY_LUT_PATH
        ),
        distance_km=ONE_AU_IN_KM,
        psi=_INFLOW_PSI_DEG,
    )

    speed_in_sw_frame = np.linspace(1.0, 1.15 * _CUTOFF_SPEED_KMS, 1000)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for cooling_index in _COOLING_INDICES:
        params = FittingParameters(
            cooling_index=cooling_index,
            ionization_rate=_IONIZATION_RATE_HZ,
            cutoff_speed=_CUTOFF_SPEED_KMS,
            background_count_rate=0.0,
        )
        with np.errstate(all="ignore"):
            f_pui = np.asarray(
                distribution.f(speed_in_sw_frame, params, apply_cutoff=True),
                dtype=float,
            )
        f_pui[~np.isfinite(f_pui)] = 0.0
        f_pui[f_pui <= 0.0] = np.nan
        axis.plot(
            speed_in_sw_frame,
            f_pui,
            label=rf"$\alpha_\mathrm{{PUI}} = {cooling_index}$",
        )

    axis.axvline(
        _CUTOFF_SPEED_KMS,
        color="0.5",
        ls="--",
        lw=1,
        label=r"cutoff speed $v_b$",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"Comoving speed $v' = \|\mathbf{v} - \mathbf{v}_\text{sw}\|$ [km/s]")
    axis.set_ylabel(r"$f_\text{PUI}(v')$ [s$^3$ km$^{-6}$]")
    axis.set_xlim(speed_in_sw_frame[0], speed_in_sw_frame[-1])
    axis.set_ylim(1e0, 1e4)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "pui_distribution.svg"
    figure.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
