import tempfile
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import List

import imap_data_access
import numpy as np
from spacepy.pycdf import CDF
from uncertainties.unumpy import nominal_values, uarray

from imap_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.models import SwapiL2Data, SwapiL3Data
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import extract_coarse_sweep, \
    calculate_proton_solar_wind_speed


EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME = "proton_sw_speed"


class SwapiL3AProcessor:

    def __init__(self, dependencies: List[UpstreamDataDependency], instrument: str, level: str, start_date: datetime, end_date: datetime,
                 version: str):
        self.instrument = instrument
        self.level = level
        self.version = version
        self.end_date = end_date
        self.start_date = start_date
        self.dependencies = dependencies

    def download_upstream_dependencies(self) -> Path:
        dependencies = [d for d in self.dependencies if
                        d.instrument == "swapi" and d.data_level == "l2"] # and d.start_date == self.start_date] # and d.end_date == self.end_date]

        if len(dependencies) != 1:
            raise ValueError(f"Unexpected dependencies found for SWAPI L3:"
                    f"{dependencies}. Expected only one dependency.")
        files_to_download = [result['file_path'] for result in imap_data_access.query(instrument=dependencies[0].instrument,
                                                                          data_level=dependencies[0].data_level,
                                                                          descriptor=dependencies[0].descriptor,
                                                                          start_date=self.start_date.strftime("%Y%d%m"),
                                                                          end_date=self.end_date.strftime("%Y%d%m")
                                                                          )]

        if len(files_to_download) != 1:
            raise ValueError(f"Unexpected files found for SWAPI L3:"
                    f"{files_to_download}. Expected only one file to download.")

        return imap_data_access.download(files_to_download[0])

    def read_l2_swapi_data(self, cdf_path: Path) -> SwapiL2Data:
        cdf = CDF(str(cdf_path))
        return SwapiL2Data(cdf.raw_var("epoch")[...],
                           cdf["energy"][...],
                           cdf["swp_coin_rate"][...],
                           cdf["spin_angles"][...],
                           cdf["swp_coin_unc"][...])

    def chunk_l2_data(self, data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
        i = 0
        while i < len(data.epoch):
            yield SwapiL2Data(
                data.epoch[i:i+chunk_size],
                data.energy,
                data.coincidence_count_rate[i:i+chunk_size],
                data.spin_angles[i:i+chunk_size],
                data.coincidence_count_rate_uncertainty[i:i+chunk_size]
            )
            i += chunk_size

    def process(self):
        downloaded_file_path = self.download_upstream_dependencies()

        data = self.read_l2_swapi_data(downloaded_file_path)

        epochs = []
        proton_solar_wind_speeds = []
        for data_chunk in self.chunk_l2_data(data, 5):
            proton_solar_wind_speed, a, phi, b = calculate_proton_solar_wind_speed(
                uarray(data_chunk.coincidence_count_rate, data_chunk.coincidence_count_rate_uncertainty),
                data_chunk.spin_angles, data_chunk.energy, data_chunk.epoch)
            proton_solar_wind_speeds.append(proton_solar_wind_speed)
            epochs.append(data_chunk.epoch[0]+30_000_000_000)

        l3_data = SwapiL3Data(np.array(epochs), np.array(proton_solar_wind_speeds))
        l3_cdf_file_name = f'imap_{self.instrument}_{self.level}_fake-menlo-{uuid.uuid4()}_{self.start_date.strftime("%Y%d%m")}_{self.version}.cdf'
        l3_cdf_file_path = f'{TEMP_CDF_FOLDER_PATH}/{l3_cdf_file_name}'
        l3_cdf = CDF(l3_cdf_file_path, '')
        l3_cdf[EPOCH_CDF_VAR_NAME] = l3_data.epoch
        l3_cdf[PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME] = nominal_values(l3_data.proton_sw_speed)

        l3_cdf.new(EPOCH_DELTA_CDF_VAR_NAME, 30_000_000_000, recVary=False)
        l3_cdf.close()
        imap_data_access.upload(l3_cdf_file_path)

        return
