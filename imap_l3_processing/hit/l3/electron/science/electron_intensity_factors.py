import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ELECTRON_ENERGY_BIN_COUNT = 6
EXPECTED_HEADER = (
    "IALIRT_Rate_A_side",
    "IALIRT_Rate_B_side",
    "delta_E_MeV",
    "Gfavg_cm2_sr",
    "b",
)


@dataclass
class ElectronIntensityFactors:
    a_side_rate_indices: np.ndarray
    b_side_rate_indices: np.ndarray
    delta_e_mev: np.ndarray
    geometry_factor_cm2_sr: np.ndarray
    background: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> "ElectronIntensityFactors":
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            rows = list(reader)
        if tuple(header) != EXPECTED_HEADER:
            raise ValueError(f"Unexpected header in {path}: {','.join(header)}")
        if len(rows) != ELECTRON_ENERGY_BIN_COUNT:
            raise ValueError(
                f"Expected {ELECTRON_ENERGY_BIN_COUNT} energy bins in {path}: got {len(rows)}"
            )
        columns = np.array(rows, dtype=float)
        return cls(
            a_side_rate_indices=(columns[:, 0] - 1).astype(int),
            b_side_rate_indices=(columns[:, 1] - 1).astype(int),
            delta_e_mev=columns[:, 2],
            geometry_factor_cm2_sr=columns[:, 3],
            background=columns[:, 4],
        )

    @property
    def energy_upper_mev(self) -> np.ndarray:
        return np.cumsum(self.delta_e_mev)

    @property
    def energy_lower_mev(self) -> np.ndarray:
        return self.energy_upper_mev - self.delta_e_mev

    @property
    def energy_labels(self) -> list[str]:
        lower_kev = [round(float(lower_mev) * 1000) for lower_mev in self.energy_lower_mev]
        upper_kev = [round(float(upper_mev) * 1000) for upper_mev in self.energy_upper_mev]
        labels = [f"{lower}-{upper} keV" for lower, upper in zip(lower_kev, upper_kev)]
        labels[0] = f"<{upper_kev[0]} keV"
        labels[-1] = f">{lower_kev[-1]} keV"
        return labels
