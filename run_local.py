import sys
from datetime import datetime
from pathlib import Path

from bitstring import BitStream
from spacepy.pycdf import CDF

from imap_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR
from imap_processing.glows.glows_processor import GlowsProcessor
from imap_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_processing.glows.l3a.utils import read_l2_glows_data
from imap_processing.hit.l3.pha.pha_event_reader import PHAEventReader
from imap_processing.hit.l3.pha.science.calculate_pha import process_pha_event
from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import CosineCorrectionLookupTable
from imap_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable
from imap_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup
from imap_processing.models import InputMetadata, UpstreamDataDependency
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
from imap_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_processing.swe.swe_processor import SweProcessor
from imap_processing.utils import save_data
from tests.test_helpers import get_test_data_path


def create_glows_l3a_cdf(dependencies: GlowsL3ADependencies):
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3a',
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2010, 1, 2),
        version='v001')

    upstream_dependencies = [
        UpstreamDataDependency(input_metadata.instrument,
                               "l2",
                               input_metadata.start_date,
                               input_metadata.end_date,
                               input_metadata.version,
                               GLOWS_L2_DESCRIPTOR)
    ]
    processor = GlowsProcessor(upstream_dependencies, input_metadata)

    l3a_data = processor.process_l3a(dependencies)
    cdf_path = save_data(l3a_data)
    return cdf_path


def create_swapi_l3b_cdf(geometric_calibration_file, efficiency_calibration_file, cdf_file):
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


def create_swapi_l3a_cdf(proton_temperature_density_calibration_file, alpha_temperature_density_calibration_file,
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


def create_swe_cdf(dependencies: SweL3Dependencies)->str:
    input_metadata = InputMetadata(
        instrument='swe',
        data_level='l3',
        start_date=datetime(2025, 10, 23),
        end_date=datetime(2025, 10, 24),
        version='v999')
    processor = SweProcessor(None, input_metadata)
    output_data =processor.calculate_pitch_angle_products(dependencies)
    cdf_path = save_data(output_data)
    return cdf_path

def process_hit_pha():
    bitstream = BitStream(filename=get_test_data_path("hit/pha_events/full_event_record_buffer.bin"))
    events = PHAEventReader.read_all_pha_events(bitstream.bin)

    cosine_table = CosineCorrectionLookupTable(
        get_test_data_path("hit/pha_events/imap_hit_l3_r2-cosines-text-not-cdf_20250203_v001.cdf"),
        get_test_data_path("hit/pha_events/imap_hit_l3_r3-cosines-text-not-cdf_20250203_v001.cdf"),
        get_test_data_path("hit/pha_events/imap_hit_l3_r4-cosines-text-not-cdf_20250203_v001.cdf"),
    )
    gain_table = GainLookupTable.from_file(
        get_test_data_path("hit/pha_events/imap_hit_l3_high-gains-text-not-cdf_20250203_v001.cdf"),
        get_test_data_path("hit/pha_events/imap_hit_l3_low-gains-text-not-cdf_20250203_v001.cdf"))

    range_fit_lookup = RangeFitLookup.from_files(
        get_test_data_path("hit/pha_events/imap_hit_l3_range2-fit-text-not-cdf_20250203_v001.cdf"),
        get_test_data_path("hit/pha_events/imap_hit_l3_range3-fit-text-not-cdf_20250203_v001.cdf"),
        get_test_data_path("hit/pha_events/imap_hit_l3_range4-fit-text-not-cdf_20250203_v001.cdf"),
    )
    processed_events = [process_pha_event(e, cosine_table, gain_table, range_fit_lookup) for e in events]
    print(processed_events)


if __name__ == "__main__":
    if "swapi" in sys.argv:
        if "l3a" in sys.argv:
            paths = create_swapi_l3a_cdf(
                "tests/test_data/swapi/imap_swapi_l2_density-temperature-lut-text-not-cdf_20240905_v002.cdf",
                "tests/test_data/swapi/imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf",
                "tests/test_data/swapi/imap_swapi_l2_clock-angle-and-flow-deflection-lut-text-not-cdf_20240918_v001.cdf",
                "tests/test_data/swapi/imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v002.cdf",
                "tests/test_data/swapi/imap_swapi_l2_instrument-response-lut-zip-not-cdf_20241023_v001.cdf",
                "tests/test_data/swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf",
                "tests/test_data/swapi/imap_swapi_l2_50-sweeps_20250606_v001.cdf"
            )
            print(paths)
        if "l3b" in sys.argv:
            path = create_swapi_l3b_cdf(
                "tests/test_data/swapi/imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v002.cdf",
                "tests/test_data/swapi/imap_swapi_l2_efficiency-lut-text-not-cdf_20241020_v003.cdf",
                "tests/test_data/swapi/imap_swapi_l2_sci_20100101_v001.cdf")
            print(path)
    if "glows" in sys.argv:
        cdf_data = CDF("tests/test_data/glows/imap_glows_l2_hist_20130908_v003.cdf")
        l2_glows_data = read_l2_glows_data(cdf_data)

        dependencies = GlowsL3ADependencies(l2_glows_data, {
            "calibration_data": Path(
                "instrument_team_data/glows/imap_glows_l3a_calibration-data-text-not-cdf_20250707_v002.cdf"),
            "settings": Path(
                "instrument_team_data/glows/imap_glows_l3a_pipeline-settings-json-not-cdf_20250707_v002.cdf"),
            "time_dependent_bckgrd": Path(
                "instrument_team_data/glows/imap_glows_l3a_time-dep-bckgrd-text-not-cdf_20250707_v001.cdf"),
            "extra_heliospheric_bckgrd": Path(
                "instrument_team_data/glows/imap_glows_l3a_map-of-extra-helio-bckgrd-text-not-cdf_20250707_v001.cdf"),
        })

        path = create_glows_l3a_cdf(dependencies)
        print(path)

    if "hit" in sys.argv:
        process_hit_pha()

    if "swe" in sys.argv:
        dependencies = SweL3Dependencies.from_file_paths(
            get_test_data_path("swe/imap_swe_l2_sci_20250101_v002.cdf"),
            get_test_data_path("mag/imap_mag_l1d_mago-normal_20250101_v001.cdf"),
            get_test_data_path("swe/imap_swapi_l3a_proton-sw_20250101_v001.cdf"),
            get_test_data_path("swe/example_swe_config.json"),
        )
        print(create_swe_cdf(dependencies))

