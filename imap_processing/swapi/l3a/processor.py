import uuid
from datetime import datetime, date
from pathlib import Path
from typing import List

import imap_data_access
import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import uarray

from imap_processing.cdf.cdf_utils import write_cdf
from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, TEMP_CDF_FOLDER_PATH
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData, SwapiL3AlphaSolarWindData
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_alpha_solar_wind_speed
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_proton_solar_wind_speed
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    TemperatureAndDensityCalibrationTable, calculate_proton_solar_wind_temperature_and_density
from imap_processing.swapi.l3a.utils import read_l2_swapi_data, chunk_l2_data


class SwapiL3AProcessor:

    def __init__(self, dependencies: List[UpstreamDataDependency], instrument: str, level: str, start_date: datetime,
                 end_date: datetime,
                 version: str):
        self.instrument = instrument
        self.level = level
        self.version = version
        self.end_date = end_date
        self.start_date = start_date
        self.dependencies = dependencies

    def download_upstream_dependencies(self) -> Path:
        dependencies = [d for d in self.dependencies if
                        d.instrument == "swapi" and d.data_level == "l2"]  # and d.start_date == self.start_date] # and d.end_date == self.end_date]

        if len(dependencies) != 1:
            raise ValueError(f"Unexpected dependencies found for SWAPI L3:"
                             f"{dependencies}. Expected only one dependency.")
        files_to_download = [result['file_path'] for result in
                             imap_data_access.query(instrument=dependencies[0].instrument,
                                                    data_level=dependencies[0].data_level,
                                                    descriptor=dependencies[0].descriptor,
                                                    start_date=self.start_date.strftime("%Y%d%m"),
                                                    end_date=self.end_date.strftime("%Y%d%m"),
                                                    version='latest'
                                                    )]
        if len(files_to_download) != 1:
            raise ValueError(f"Unexpected files found for SWAPI L3:"
                             f"{files_to_download}. Expected only one file to download.")

        return imap_data_access.download(files_to_download[0])

    def download_calibration_tables(self) -> Path:
        density_temperature_files = [result["file_path"] for result in imap_data_access.query(instrument="swapi",
                                                                                              data_level="l3a",
                                                                                              descriptor="density-temperature-lut-text-not-cdf",
                                                                                              version='latest')]
        if len(density_temperature_files) != 1:
            raise ValueError(f"Unexpected files found for SWAPI L3 density temperature calibration file query:"
                             f"{density_temperature_files}. Expected only one file to download.")

        return imap_data_access.download(density_temperature_files[0])

    def process(self):
        downloaded_file_path = self.download_upstream_dependencies()
        density_temperature_calibration_file = self.download_calibration_tables()
        temperature_and_density_calibrator = TemperatureAndDensityCalibrationTable.from_file(
            density_temperature_calibration_file)

        data = read_l2_swapi_data(downloaded_file_path)

        epochs = []

        proton_solar_wind_speeds = []
        proton_solar_wind_temperatures = []
        proton_solar_wind_density = []

        alpha_solar_wind_speeds = []

        for data_chunk in chunk_l2_data(data, 5):
            coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                              data_chunk.coincidence_count_rate_uncertainty)
            proton_solar_wind_speed, a, phi, b = calculate_proton_solar_wind_speed(
                coincidence_count_rates_with_uncertainty,
                data_chunk.spin_angles, data_chunk.energy, data_chunk.epoch)
            proton_solar_wind_speeds.append(proton_solar_wind_speed)

            temperature, density = calculate_proton_solar_wind_temperature_and_density(
                temperature_and_density_calibrator,
                proton_solar_wind_speed,
                ufloat(0.0, 1.0),
                phi,
                coincidence_count_rates_with_uncertainty,
                data_chunk.energy)

            proton_solar_wind_temperatures.append(temperature)
            proton_solar_wind_density.append(density)

            epochs.append(data_chunk.epoch[0] + THIRTY_SECONDS_IN_NANOSECONDS)

            alpha_solar_wind_speeds.append(calculate_alpha_solar_wind_speed(
                coincidence_count_rates_with_uncertainty,
                data_chunk.energy
            ))

        proton_solar_wind_l3_data = SwapiL3ProtonSolarWindData(np.array(epochs), np.array(proton_solar_wind_speeds),
                                                               np.array(proton_solar_wind_temperatures),
                                                               np.array(proton_solar_wind_density))

        formatted_start_date = self.start_date.strftime("%Y%d%m")
        logical_file_id = f'imap_{self.instrument}_{self.level}_proton-sw-speed-fake-menlo-{uuid.uuid4()}_{formatted_start_date}_{self.version}'
        file_path = f'{TEMP_CDF_FOLDER_PATH}/{logical_file_id}.cdf'
        attribute_manager = ImapAttributeManager()
        attribute_manager.add_global_attribute("Data_version", self.version)
        attribute_manager.add_instrument_attrs(self.instrument, self.level)
        attribute_manager.add_global_attribute("Generation_date", date.today().strftime("%Y%m%d"))
        attribute_manager.add_global_attribute("Logical_source", 'imap_swapi_l3a_proton-sw-speed')
        attribute_manager.add_global_attribute("Logical_file_id", logical_file_id)
        write_cdf(file_path, proton_solar_wind_l3_data, attribute_manager)
        imap_data_access.upload(file_path)

        alpha_solar_wind_l3_data = SwapiL3AlphaSolarWindData(np.array(epochs), np.array(alpha_solar_wind_speeds))
        logical_file_id = f'imap_{self.instrument}_{self.level}_alpha-sw-speed-fake-menlo-{uuid.uuid4()}_{formatted_start_date}_{self.version}'
        attribute_manager.add_global_attribute("Logical_source", 'imap_swapi_l3a_alpha-sw-speed')
        attribute_manager.add_global_attribute("Logical_file_id", logical_file_id)
        file_path = f'{TEMP_CDF_FOLDER_PATH}/{logical_file_id}.cdf'
        write_cdf(file_path, alpha_solar_wind_l3_data, attribute_manager)
        imap_data_access.upload(file_path)
