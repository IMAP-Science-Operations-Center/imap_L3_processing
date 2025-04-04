from dataclasses import dataclass
from pathlib import Path

from imap_l3_processing.codice.l2.direct_event.science.azimuth_lookup import AzimuthLookup
from imap_l3_processing.codice.l2.direct_event.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l2.direct_event.science.time_of_flight_lookup import TimeOfFlightLookup
from imap_l3_processing.codice.models import CodiceL1aHiData
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency


@dataclass
class CodiceL2Dependencies:
    codice_l1a_hi_data: CodiceL1aHiData
    energy_lookup_table: EnergyLookup
    azimuth_lookup_table: AzimuthLookup
    time_of_flight_lookup_table: TimeOfFlightLookup

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        try:
            l1_cdf = next(d for d in dependencies if
                          d.instrument == "codice" and d.data_level == "l1a" and d.descriptor == "direct-events")
        except StopIteration as e:
            raise e

        l1a_cdf_file_path = download_dependency(l1_cdf)
        energy_file_path = download_dependency(cls.create_ancillary_dependency("energy-lookup"))
        energy_bin_file_path = download_dependency(cls.create_ancillary_dependency("energy-bin-lookup"))
        tof_file_path = download_dependency(cls.create_ancillary_dependency("time-of-flight-lookup"))
        azimuth_file_path = download_dependency(cls.create_ancillary_dependency("azimuth-lookup"))

        return cls.from_file_paths(l1a_cdf_file_path, energy_file_path, energy_bin_file_path, tof_file_path,
                                   azimuth_file_path)

    @classmethod
    def from_file_paths(cls, codice_hi_l1_cdf_path, energy_lookup_file: Path, energy_bin_lookup_file: Path,
                        time_of_flight_lookup_file: Path,
                        azimuth_lookup_file: Path):
        energy_lookup = EnergyLookup.from_files(energy_lookup_file, energy_bin_lookup_file)
        time_of_flight_lookup = TimeOfFlightLookup.from_files(time_of_flight_lookup_file)
        azimuth_lookup = AzimuthLookup.from_files(azimuth_lookup_file)
        codice_hi_l1_cdf = CodiceL1aHiData.read_from_cdf(codice_hi_l1_cdf_path)

        return cls(codice_hi_l1_cdf, energy_lookup, azimuth_lookup, time_of_flight_lookup)

    @classmethod
    def create_ancillary_dependency(cls, descriptor: str):
        return UpstreamDataDependency(
            descriptor=descriptor,
            instrument="codice",
            data_level="l2",
            start_date=None,
            end_date=None,
            version="latest",
        )
