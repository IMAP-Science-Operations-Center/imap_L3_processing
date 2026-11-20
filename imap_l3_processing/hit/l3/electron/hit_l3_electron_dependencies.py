from dataclasses import dataclass

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.hit.l3.electron.models import HitL1bIalirtRates
from imap_l3_processing.hit.l3.electron.science.electron_intensity_factors import (
    ElectronIntensityFactors,
)

HIT_L1B_STANDARD_RATES_DESCRIPTOR = "standard-rates"
HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR = "electron-intensity-factors"


@dataclass
class HitL3ElectronDependencies:
    ialirt_rates: HitL1bIalirtRates
    intensity_factors: ElectronIntensityFactors

    @classmethod
    def fetch_dependencies(
        cls, dependencies: ProcessingInputCollection
    ) -> "HitL3ElectronDependencies":
        l1b_files = dependencies.get_file_paths(
            source="hit",
            data_type="l1b",
            descriptor=HIT_L1B_STANDARD_RATES_DESCRIPTOR,
        )
        if not l1b_files:
            raise ValueError(f"Missing {HIT_L1B_STANDARD_RATES_DESCRIPTOR} dependency.")
        factor_files = dependencies.get_file_paths(
            source="hit",
            descriptor=HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR,
        )
        if not factor_files:
            raise ValueError(f"Missing {HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR} dependency.")

        l1b_path = imap_data_access.download(l1b_files[0])
        factors_path = imap_data_access.download(factor_files[0])

        return cls(
            ialirt_rates=HitL1bIalirtRates.read_from_cdf(l1b_path),
            intensity_factors=ElectronIntensityFactors.from_file(factors_path),
        )
