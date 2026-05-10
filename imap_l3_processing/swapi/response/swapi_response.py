from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
)
from imap_l3_processing.swapi.response.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.passband_grid import (
    PassbandGrid,
    build_passband_grid,
)


class ResponseGrid(NamedTuple):
    """V-and-species-specific instrument response evaluated at one ESA step.

    Carries one PassbandGrid per region (`sg_passband`, `oa_passband`); call
    sites pick the relevant one by region rather than threading a flag through
    the helpers. `azimuthal_transmission` is the same `AzimuthalTransmissionGrid`
    reference across all grids in a sweep — bundling it per-grid is pointer-cheap
    and keeps the integration call signature compact."""

    sg_passband: PassbandGrid
    oa_passband: PassbandGrid
    central_speed: float
    central_effective_area: float
    azimuthal_transmission: AzimuthalTransmissionGrid


@dataclass
class SwapiResponse:
    azimuthal_transmission: (
        NDArray  # shape (N,), evenly spaced at 0.1 deg intervals from 0
    )
    central_effective_area_at_voltage: Callable  # ESA voltage [V] -> effective area [cm^2], piecewise linear, clamped at endpoints
    passband_fit_coefficients: (
        pd.DataFrame
    )  # index: (region, energy_ratio, elevation), columns: [2, 1, 0]
    passband_esa_voltage_limits: dict  # {region: (min_esa_voltage, max_esa_voltage)}
    _grid_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _response_grid_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # Azimuthal transmission table is sampled at 0.1 deg spacing; constant for all V/species.
    AZIMUTHAL_TRANSMISSION_SPACING_DEG = 0.1

    @classmethod
    def from_files(
        cls,
        azimuthal_transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
    ) -> "SwapiResponse":
        transmission_df = pd.read_csv(azimuthal_transmission_path)
        area_df = pd.read_csv(central_effective_area_path)
        coeffs_df = pd.read_csv(
            passband_fit_coefficients_path,
            index_col=["region", "energy_ratio", "elevation"],
        )
        limit_cols = ["min_esa_voltage", "max_esa_voltage"]
        limits_df = coeffs_df[limit_cols]
        coeffs_df = coeffs_df.drop(columns=limit_cols)
        esa_limits = {
            region: (
                float(limits_df.xs(region, level="region")["min_esa_voltage"].iloc[0]),
                float(limits_df.xs(region, level="region")["max_esa_voltage"].iloc[0]),
            )
            for region in limits_df.index.get_level_values("region").unique()
        }
        coeffs_df.columns = coeffs_df.columns.astype(int)

        central_effective_area_values = area_df["effective_area"].values
        return cls(
            azimuthal_transmission=transmission_df["transmission"].fillna(0).values,
            central_effective_area_at_voltage=interp1d(
                area_df["esa_voltage"].values,
                central_effective_area_values,
                kind="linear",
                bounds_error=False,
                fill_value=(
                    float(central_effective_area_values[0]),
                    float(central_effective_area_values[-1]),
                ),
                assume_sorted=True,
            ),
            passband_fit_coefficients=coeffs_df,
            passband_esa_voltage_limits=esa_limits,
        )

    def get_central_effective_area(self, esa_voltage: float) -> float:
        return float(self.central_effective_area_at_voltage(np.abs(esa_voltage)))

    def central_speed(
        self, esa_voltage: float, mass_per_charge_m_p_per_e: float
    ) -> float:
        """Central proton-frame speed (km/s) at the given ESA voltage for a species
        whose mass-per-charge is `mass_per_charge_m_p_per_e` (1 for proton, ≈2 for alpha):
            v_0 = sqrt(2 k* |V| (e/m_p) / mass_per_charge_m_p_per_e).
        """
        from imap_l3_processing.constants import (
            METERS_PER_KILOMETER,
            PROTON_CHARGE_OVER_MASS_C_PER_KG,
        )

        return float(
            np.sqrt(
                2.0
                * SWAPI_K_FACTOR
                * abs(esa_voltage)
                * PROTON_CHARGE_OVER_MASS_C_PER_KG
                / float(mass_per_charge_m_p_per_e)
            )
            / METERS_PER_KILOMETER
        )

    def warm_cache(self, esa_voltages) -> None:
        """Build and cache (sg_passband, oa_passband) tuples for every unique finite
        voltage in `esa_voltages`.

        This is the only path that builds grids. Call this in the parent process before
        forking workers so the ~1.8 ms pandas pivot is paid once per voltage and forked
        workers inherit the populated cache via COW rather than rebuilding independently.
        Calling with a voltage already in the cache is a no-op.
        """
        for v in np.unique(np.asarray(esa_voltages, dtype=float).ravel()):
            key = self._cache_key(v)
            if np.isfinite(v) and key not in self._grid_cache:
                self._grid_cache[key] = dict()
                for region in ('SG', 'OA'):
                    passband = build_passband_grid(self._get_passband_values(v, region))
                    self._grid_cache[key][region] = passband

    def create_response_grid(
        self,
        esa_voltage: float,
        mass_per_charge_m_p_per_e: float,
        central_effective_area_scale: float = 1.0,
    ) -> ResponseGrid:
        """Return a per-voltage `ResponseGrid`, cached by `(voltage, species, ea_scale)`.

        Cached so that repeated fits at the same voltage (e.g. every LM iter in a chunk)
        don't pay the ResponseGrid build cost. Cache hit is an O(1) dict lookup.
        """
        cache_key = (
            self._cache_key(float(esa_voltage)),
            self._cache_key(float(mass_per_charge_m_p_per_e)),
            self._cache_key(float(central_effective_area_scale)),
        )
        cached = self._response_grid_cache.get(cache_key)
        if cached is not None:
            return cached
        sg_passband = self.create_passband_grid(esa_voltage, "SG")
        oa_passband = self.create_passband_grid(esa_voltage, "OA")
        response_grid = ResponseGrid(
            sg_passband=sg_passband,
            oa_passband=oa_passband,
            central_speed=self.central_speed(esa_voltage, mass_per_charge_m_p_per_e),
            central_effective_area=(
                self.get_central_effective_area(esa_voltage)
                * float(central_effective_area_scale)
            ),
            azimuthal_transmission=AzimuthalTransmissionGrid(
                values=np.asarray(self.azimuthal_transmission, dtype=float),
                spacing=float(self.AZIMUTHAL_TRANSMISSION_SPACING_DEG),
            ),
        )
        self._response_grid_cache[cache_key] = response_grid
        return response_grid

    def create_passband_grid(self, esa_voltage: float, region: str) -> PassbandGrid:
        """Return the cached PassbandGrid for `(esa_voltage, region)`.

        `region` is `"SG"` or `"OA"`. Raises KeyError if `warm_cache` was not called
        for this voltage. The cache must be populated before any call to this method —
        call `warm_cache(voltages)` first.
        """
        cache_key = self._cache_key(esa_voltage)
        try:
            passband = self._grid_cache[cache_key][region]
        except KeyError:
            raise KeyError(
                f"No PassbandGrid cached for ESA voltage {esa_voltage} V, region {region}. "
                f"Call warm_cache([{esa_voltage}]) before create_passband_grid."
            ) from None
        return passband


    def _get_passband_values(self, esa_voltage: float, region: str) -> pd.DataFrame:
        v_min, v_max = self.passband_esa_voltage_limits.get(region, (0, np.inf))
        clamped_voltage = float(np.clip(np.abs(esa_voltage), v_min, v_max))
        log_beam_energy = np.log(SWAPI_K_FACTOR * clamped_voltage)
        coeffs = self.passband_fit_coefficients.xs(region, level="region")
        values = np.exp(np.polyval(coeffs.values.T, log_beam_energy))
        return pd.DataFrame(values, index=coeffs.index, columns=["value"])

    def _cache_key(self, x) -> float:
        return round(float(x), 3)
