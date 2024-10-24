import sys
from datetime import datetime

from spacepy.pycdf import CDF

from imap_processing import spice_wrapper
from imap_processing.models import InputMetadata
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable
from imap_processing.swapi.l3a.science.calculate_pickup_ion import DensityOfNeutralHeliumLookupTable
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    ClockAngleCalibrationTable
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    ProtonTemperatureAndDensityCalibrationTable
from imap_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_processing.swapi.l3a.utils import read_l2_swapi_data
from imap_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTableCollection
from imap_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_processing.swapi.swapi_processor import SwapiProcessor
from imap_processing.utils import save_data


def create_l3b_cdf(geometric_calibration_file, efficiency_calibration_file, cdf_file):
    geometric_calibration = GeometricFactorCalibrationTable.from_file(geometric_calibration_file)
    efficiency_calibration = EfficiencyCalibrationTable(efficiency_calibration_file)
    cdf_data = CDF(cdf_file)
    swapi_l3_dependencies = SwapiL3BDependencies(cdf_data, geometric_calibration, efficiency_calibration)
    swapi_data = read_l2_swapi_data(swapi_l3_dependencies.data)

    input_metadata = InputMetadata(
        instrument='swapi',
        data_level='l3b',
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2010, 1, 2),
        version='v999')
    processor = SwapiProcessor(None, input_metadata)

    l3b_combined_vdf = processor.process_l3b(swapi_data, swapi_l3_dependencies)
    cdf_path = save_data(l3b_combined_vdf)
    return cdf_path


def create_l3a_cdf(proton_temperature_density_calibration_file, alpha_temperature_density_calibration_file,
                   clock_angle_and_flow_deflection_calibration_file, geometric_factor_calibration_file,
                   instrument_response_calibration_file, density_of_neutral_helium_calibration_file,
                   cdf_file):
    proton_temperature_density_calibration_table = ProtonTemperatureAndDensityCalibrationTable.from_file(
        proton_temperature_density_calibration_file)
    alpha_temperature_density_calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(
        alpha_temperature_density_calibration_file)
    clock_angle_and_flow_deflection_calibration_table = ClockAngleCalibrationTable.from_file(
        clock_angle_and_flow_deflection_calibration_file)
    geometric_factor_calibration_table = GeometricFactorCalibrationTable.from_file(geometric_factor_calibration_file)
    instrument_response_calibration_table = InstrumentResponseLookupTableCollection.from_file(
        instrument_response_calibration_file)
    density_of_neutral_helium_calibration_table = DensityOfNeutralHeliumLookupTable.from_file(
        density_of_neutral_helium_calibration_file)
    cdf_data = CDF(cdf_file)
    swapi_l3_dependencies = SwapiL3ADependencies(cdf_data, proton_temperature_density_calibration_table,
                                                 alpha_temperature_density_calibration_table,
                                                 clock_angle_and_flow_deflection_calibration_table,
                                                 geometric_factor_calibration_table,
                                                 instrument_response_calibration_table,
                                                 density_of_neutral_helium_calibration_table)
    swapi_data = read_l2_swapi_data(swapi_l3_dependencies.data)

    input_metadata = InputMetadata(
        instrument='swapi',
        data_level='l3a',
        start_date=datetime(2025, 10, 23),
        end_date=datetime(2025, 10, 24),
        version='v999')
    processor = SwapiProcessor(None, input_metadata)

    l3a_proton_sw, l3a_alpha_sw, l3a_pui_he = processor.process_l3a(swapi_data, swapi_l3_dependencies)
    proton_cdf_path = save_data(l3a_proton_sw)
    alpha_cdf_path = save_data(l3a_alpha_sw)
    pui_he_cdf_path = save_data(l3a_pui_he)
    return proton_cdf_path, alpha_cdf_path, pui_he_cdf_path


if __name__ == "__main__":
    if "l3a" in sys.argv:
        spice_wrapper.furnish()
        paths = create_l3a_cdf(
            "swapi/test_data/imap_swapi_l2_density-temperature-lut-text-not-cdf_20240905_v002.cdf",
            "swapi/test_data/imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf",
            "swapi/test_data/imap_swapi_l2_clock-angle-and-flow-deflection-lut-text-not-cdf_20240918_v001.cdf",
            "swapi/test_data/imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v002.cdf",
            "swapi/test_data/imap_swapi_l2_instrument-response-lut-zip-not-cdf_20241023_v001.cdf",
            "swapi/test_data/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v001.cdf",
            "swapi/test_data/imap_swapi_l2_50-sweeps_20250606_v001.cdf"
        )
        print(paths)
    if "l3b" in sys.argv:
        path = create_l3b_cdf(
            "swapi/test_data/imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v002.cdf",
            "swapi/test_data/imap_swapi_l2_efficiency-lut-text-not-cdf_20241020_v003.cdf",
            "swapi/test_data/imap_swapi_l2_sci_20100101_v001.cdf")
        print(path)
