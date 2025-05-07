from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TypeVar
from unittest.mock import patch, Mock

import imap_data_access
import numpy as np
import xarray as xr
from imap_data_access.processing_input import AncillaryInput, ProcessingInputCollection, ScienceInput
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2SWSpeciesData, CodiceLoL2DirectEventData, \
    CodiceLoL1aSWPriorityRates, CodiceLoL1aNSWPriorityRates
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.glows.glows_initializer import GlowsInitializer
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.utils import read_l2_glows_data, create_glows_l3a_dictionary_from_cdf
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies import HiL3CombinedMapDependencies
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.hi.l3.models import HiL3SpectralIndexDataProduct, HiL3IntensityDataProduct, combine_maps
from imap_l3_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_l3_processing.hit.l3.hit_processor import HitProcessor
from imap_l3_processing.hit.l3.models import HitL1Data
from imap_l3_processing.hit.l3.pha.hit_l3_pha_dependencies import HitL3PhaDependencies
from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import CosineCorrectionLookupTable
from imap_l3_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable
from imap_l3_processing.hit.l3.pha.science.hit_event_type_lookup import HitEventTypeLookup
from imap_l3_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup
from imap_l3_processing.hit.l3.utils import read_l2_hit_data
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    ClockAngleCalibrationTable
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    ProtonTemperatureAndDensityCalibrationTable
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import read_l2_swapi_data
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import \
    InstrumentResponseLookupTableCollection
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.swe.swe_processor import SweProcessor
from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData, UltraL2Map
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies
from imap_l3_processing.ultra.l3.ultra_processor import UltraProcessor
from imap_l3_processing.utils import save_data, read_l1d_mag_data
from scripts.hi.create_hi_full_spin_deps import create_hi_full_spin_deps
from scripts.ultra.create_example_ultra_l1c_pset import _write_ultra_l1c_cdf_with_parents
from scripts.ultra.create_example_ultra_l2_map import _write_ultra_l2_cdf_with_parents
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path, environment_variables, \
    try_get_many_run_local_paths


def create_glows_l3a_cdf(dependencies: GlowsL3ADependencies):
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3a',
        start_date=datetime(2013, 9, 8),
        end_date=datetime(2013, 9, 8),
        version='v001')

    processor = GlowsProcessor(Mock(), input_metadata)

    l3a_data = processor.process_l3a(dependencies)
    cdf_path = save_data(l3a_data, delete_if_present=True)
    return cdf_path


def create_codice_lo_l3a_partial_densities_cdf():
    codice_lo_l2_data = CodiceLoL2SWSpeciesData.read_from_cdf(
        get_test_instrument_team_data_path('codice/lo/imap_codice_l2_lo-sw-species_20241110193900_v0.0.2.cdf'))
    mpc_lookup = MassPerChargeLookup.read_from_file(get_test_data_path('codice/test_mass_per_charge_lookup.csv'))
    deps = CodiceLoL3aPartialDensitiesDependencies(codice_l2_lo_data=codice_lo_l2_data,
                                                   mass_per_charge_lookup=mpc_lookup)

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='lo-partial-densities'
    )

    codice_lo_processor = CodiceLoProcessor(ProcessingInputCollection(), input_metadata)
    partial_densities_data = codice_lo_processor.process_l3a_partial_densities(deps)
    cdf_path = save_data(partial_densities_data, delete_if_present=True)
    return cdf_path


def create_codice_lo_l3a_direct_events_cdf():
    codice_lo_l2_direct_events = CodiceLoL2DirectEventData.read_from_cdf(
        get_test_instrument_team_data_path('codice/lo/imap_codice_l2_lo-direct-events_20241110193700_v0.0.2.cdf'))
    codice_lo_l1a_sw_priority = CodiceLoL1aSWPriorityRates.read_from_cdf(
        get_test_instrument_team_data_path('codice/lo/imap_codice_l1a_lo-sw-priority_20241110193900_v0.0.2.cdf'))
    codice_lo_l1a_nsw_priority = CodiceLoL1aNSWPriorityRates.read_from_cdf(
        get_test_instrument_team_data_path('codice/lo/imap_codice_l1a_lo-nsw-priority_20241110193900_v0.0.2.cdf'))

    mass_coefficient_lookup = MassCoefficientLookup.read_from_csv(
        get_test_data_path('codice/mass_coefficient_lookup.csv'))

    deps = CodiceLoL3aDirectEventsDependencies(codice_l2_direct_events=codice_lo_l2_direct_events,
                                               codice_lo_l1a_sw_priority_rates=codice_lo_l1a_sw_priority,
                                               codice_lo_l1a_nsw_priority_rates=codice_lo_l1a_nsw_priority,
                                               mass_coefficient_lookup=mass_coefficient_lookup)

    input_metadata = InputMetadata(
        instrument='codice',
        data_level='l3a',
        start_date=datetime(2024, 11, 10),
        end_date=datetime(2025, 1, 2),
        version='v000',
        descriptor='lo-direct-events'
    )

    codice_lo_processor = CodiceLoProcessor(Mock(), input_metadata)
    direct_event_data = codice_lo_processor.process_l3a_direct_event_data_product(deps)
    return save_data(direct_event_data, delete_if_present=True)


def create_swapi_l3b_cdf(geometric_calibration_file, efficiency_calibration_file, cdf_file):
    geometric_calibration = GeometricFactorCalibrationTable.from_file(geometric_calibration_file)
    efficiency_calibration = EfficiencyCalibrationTable(efficiency_calibration_file)
    swapi_cdf_data = CDF(cdf_file)
    swapi_data = read_l2_swapi_data(swapi_cdf_data)

    swapi_l3_dependencies = SwapiL3BDependencies(swapi_data, geometric_calibration, efficiency_calibration)

    input_metadata = InputMetadata(
        instrument='swapi',
        data_level='l3b',
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2010, 1, 2),
        version='v000')
    processor = SwapiProcessor(Mock(), input_metadata)

    l3b_combined_vdf = processor.process_l3b(swapi_data, swapi_l3_dependencies)
    cdf_path = save_data(l3b_combined_vdf, delete_if_present=True)
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
    swapi_cdf_data = CDF(cdf_file)
    swapi_data = read_l2_swapi_data(swapi_cdf_data)
    swapi_l3_dependencies = SwapiL3ADependencies(swapi_data, proton_temperature_density_calibration_table,
                                                 alpha_temperature_density_calibration_table,
                                                 clock_angle_and_flow_deflection_calibration_table,
                                                 geometric_factor_calibration_table,
                                                 instrument_response_calibration_table,
                                                 density_of_neutral_helium_calibration_table)

    input_metadata = InputMetadata(
        instrument='swapi',
        data_level='l3a',
        start_date=datetime(2025, 10, 23),
        end_date=datetime(2025, 10, 24),
        version='v000')
    processor = SwapiProcessor(Mock(), input_metadata)

    l3a_proton_sw = processor.process_l3a_proton(swapi_data, swapi_l3_dependencies)
    l3a_alpha_sw = processor.process_l3a_alpha_solar_wind(swapi_data, swapi_l3_dependencies)
    l3a_pui_he = processor.process_l3a_pui(swapi_data, swapi_l3_dependencies)
    proton_cdf_path = save_data(l3a_proton_sw, delete_if_present=True)
    alpha_cdf_path = save_data(l3a_alpha_sw, delete_if_present=True)
    pui_he_cdf_path = save_data(l3a_pui_he, delete_if_present=True)
    return proton_cdf_path, alpha_cdf_path, pui_he_cdf_path


def create_swe_product(dependencies: SweL3Dependencies) -> str:
    input_metadata = InputMetadata(
        instrument='swe',
        data_level='l3',
        start_date=datetime(2025, 6, 29),
        end_date=datetime(2025, 7, 1),
        version='v000')
    processor = SweProcessor(Mock(), input_metadata)
    output_data = processor.calculate_products(dependencies)
    cdf_path = save_data(output_data, delete_if_present=True)
    return cdf_path


@patch("imap_l3_processing.swe.l3.science.moment_calculations.spiceypy.pxform")
def create_swe_product_with_fake_spice(dependencies: SweL3Dependencies, mock_spice_pxform) -> str:
    mock_spice_pxform.return_value = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])

    input_metadata = InputMetadata(
        instrument='swe',
        data_level='l3',
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2010, 1, 2),
        version='v000')
    processor = SweProcessor(Mock(), input_metadata)
    output_data = processor.calculate_products(dependencies)
    cdf_path = save_data(output_data, delete_if_present=True)
    return cdf_path


def create_survival_corrected_full_spin_cdf(dependencies: HiL3SingleSensorFullSpinDependencies) -> str:
    input_metadata = InputMetadata(instrument="hi",
                                   data_level="l3",
                                   start_date=datetime.now(),
                                   end_date=datetime.now() + timedelta(days=1),
                                   version="v000",
                                   descriptor="h90-ena-h-sf-sp-full-hae-4deg-6mo",
                                   )
    processor = HiProcessor(Mock(), input_metadata)
    output_data = processor.process_full_spin_single_sensor(dependencies)

    data_product = HiL3IntensityDataProduct(data=output_data, input_metadata=input_metadata)
    cdf_path = save_data(data_product, delete_if_present=True)
    return cdf_path


def create_spectral_index_cdf(dependencies: HiL3SpectralFitDependencies) -> str:
    input_metadata = InputMetadata(instrument="hi",
                                   data_level="l3",
                                   start_date=datetime.now(),
                                   end_date=datetime.now() + timedelta(days=1),
                                   version="v000",
                                   descriptor="h90-spx-h-hf-sp-full-hae-4deg-6mo",
                                   )
    processor = HiProcessor(Mock(), input_metadata)
    output_data = processor.process_spectral_fit_index(dependencies)
    data_product = HiL3SpectralIndexDataProduct(data=output_data, input_metadata=input_metadata)
    cdf_path = save_data(data_product, delete_if_present=True)
    return cdf_path


def create_hit_sectored_cdf(dependencies: HITL3SectoredDependencies) -> str:
    input_metadata = InputMetadata(
        instrument='hit',
        data_level='l3',
        descriptor='macropixel',
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 2),
        version='v000')
    processor = HitProcessor(Mock(), input_metadata)
    output_data = processor.process_pitch_angle_product(dependencies)
    cdf_path = save_data(output_data, delete_if_present=True)
    return cdf_path


def create_hit_direct_event_cdf():
    cosine_table = CosineCorrectionLookupTable(
        get_test_data_path("hit/pha_events/imap_hit_range-2A-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-2B-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3A-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3B-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4A-cosine-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4B-cosine-lookup_20250203_v000.csv"),
    )
    gain_table = GainLookupTable.from_file(
        get_test_data_path("hit/pha_events/imap_hit_hi-gain-lookup_20250203_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_lo-gain-lookup_20250203_v000.csv"))

    range_fit_lookup = RangeFitLookup.from_files(
        get_test_data_path("hit/pha_events/imap_hit_range-2A-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3A-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4A-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-2B-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-3B-charge-fit-lookup_20250319_v000.csv"),
        get_test_data_path("hit/pha_events/imap_hit_range-4B-charge-fit-lookup_20250319_v000.csv"),
    )

    event_type_look = HitEventTypeLookup.from_csv(
        get_test_data_path("hit/pha_events/imap_hit_hit-event-type-lookup_20250228_v000.csv"))

    hit_l1_data = HitL1Data.read_from_cdf(
        get_test_data_path("hit/pha_events/imap_hit_l1a_direct-events_20100105_v009.cdf"))

    direct_event_dependencies = HitL3PhaDependencies(hit_l1_data=hit_l1_data, cosine_correction_lookup=cosine_table,

                                                     gain_lookup=gain_table, range_fit_lookup=range_fit_lookup,
                                                     event_type_lookup=event_type_look)
    input_metadata = InputMetadata(
        instrument="hit",
        data_level="l3",
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=1),
        version="v001",
        descriptor="direct-events"
    )
    processor = HitProcessor(Mock(), input_metadata)

    product = processor.process_direct_event_product(direct_event_dependencies)
    file_path = save_data(product, delete_if_present=True)
    return file_path


@environment_variables({"REPOINT_DATA_FILEPATH": get_test_data_path("fake_1_day_repointing_file.csv")})
@patch('imap_l3_processing.glows.glows_initializer.query')
def run_l3b_initializer(mock_query):
    local_cdfs: list[str] = os.listdir(get_test_data_path("glows/l3a_products"))
    local_cdfs.remove('.DS_Store')

    l3a_dicts = [{'file_path': "glows/l3a_products/" + file_path,
                  'start_date': file_path.split('_')[4].split('-')[0],
                  'repointing': int(file_path.split('_')[4].split('-repoint')[1])
                  } for file_path in local_cdfs]

    mock_query.side_effect = [
        l3a_dicts, []
    ]

    bad_days_list = AncillaryInput('imap_glows_bad-days-list_20100101_v001.dat')
    waw_helio_ion = AncillaryInput('imap_glows_WawHelioIonMP_20100101_v002.json')
    uv_anisotropy = AncillaryInput('imap_glows_uv-anisotropy-1CR_20100101_v001.json')
    pipeline_settings = AncillaryInput('imap_glows_pipeline-settings-L3bc_20250707_v002.json')
    input_collection = ProcessingInputCollection(bad_days_list, waw_helio_ion, uv_anisotropy, pipeline_settings)

    GlowsInitializer.validate_and_initialize('v001', input_collection)


@environment_variables({"REPOINT_DATA_FILEPATH": get_test_data_path("fake_2_day_repointing_on_may18_file.csv")})
@patch('imap_l3_processing.glows.glows_initializer.query')
@patch('imap_l3_processing.glows.glows_processor.imap_data_access.upload')
def run_glows_l3bc_processor_and_initializer(_, mock_query):
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3b',
        start_date=datetime(2013, 9, 8),
        end_date=datetime(2013, 9, 8),
        version='v011')

    l3a_files = imap_data_access.query(instrument='glows', version=input_metadata.version, data_level='l3a',
                                       start_date='20100422', end_date='20100625')

    l3a_files_2 = imap_data_access.query(instrument='glows', version=input_metadata.version, data_level='l3a',
                                         start_date='20100922', end_date='20101123')
    mock_query.side_effect = [l3a_files + l3a_files_2, []]

    bad_days_list = AncillaryInput('imap_glows_bad-days-list_20100101_v001.dat')
    waw_helio_ion = AncillaryInput('imap_glows_WawHelioIonMP_20100101_v002.json')
    uv_anisotropy = AncillaryInput('imap_glows_uv-anisotropy-1CR_20100101_v001.json')
    pipeline_settings = AncillaryInput('imap_glows_pipeline-settings-L3bc_20250707_v002.json')
    input_collection = ProcessingInputCollection(bad_days_list, waw_helio_ion, uv_anisotropy, pipeline_settings)

    processor = GlowsProcessor(dependencies=input_collection, input_metadata=input_metadata)
    processor.process()


@patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
@patch("imap_l3_processing.glows.glows_processor.imap_data_access.upload")
@patch("imap_l3_processing.glows.glows_processor.Path")
@patch("imap_l3_processing.glows.glows_processor.run")
@patch("imap_l3_processing.glows.glows_processor.get_repoint_date_range")
def run_glows_l3e_lo_with_mocks(mock_get_repoint_date_range, _, mock_path, mock_upload,
                                mock_l3e_dependencies_class):
    mock_processing_input_collection = Mock()
    mock_processing_input_collection.get_file_paths.return_value = [Path("one path")]

    mock_l3e_dependencies: GlowsL3EDependencies = GlowsL3EDependencies(
        Path("imap_glows_l3d_solar-hist_20250501-repoint00005_v001.cdf"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/EnGridLo.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/EnGridHi.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/EnGridUltra.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/tessXYZ8.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/tessAng16.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/lyaSeriesV4_2021b.dat"),
        Path(
            "instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/solar_uv_anisotropy_NP.1.0_SP.1.0.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/speed3D.v01.Legendre.2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/density3D.v01.Legendre.2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/phion_Hydrogen_T12F107_2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/swEqtrElectrons5_2021b.dat"),
        Path("instrument_team_data/glows/GLOWS_L3d_to_L3e_processing/ionization.files.dat"),
        {"executable_dependency_paths": {
            "energy-grid-lo": "EnGridLo.dat",
            "energy-grid-hi": "EnGridHi.dat",
            "energy-grid-ultra": "EnGridUltra.dat",
            "tess-xyz-8": "tessXYZ8.dat",
            "tess-ang-16": "tessAng16.dat",
            "lya-series": "lyaSeriesV4_2021b.dat",
            "solar-uv-anistropy": "solar_uv_anisotropy_NP.1.0_SP.1.0.dat",
            "speed-3d": "speed3D.v01.Legendre.2021b.dat",
            "density-3d": "density3D.v01.Legendre.2021b.dat",
            "phion-hydrogen": "phion_Hydrogen_T12F107_2021b.dat",
            "sw-eqtr-electrons": "swEqtrElectrons5_2021b.dat",
            "ionization-files": "ionization.files.dat",
        }}
    )

    mock_l3e_dependencies.rename_dependencies = Mock()
    mock_l3e_dependencies_class.fetch_dependencies.return_value = (mock_l3e_dependencies, 5)

    mock_path.side_effect = [
        Path(get_test_instrument_team_data_path("glows/probSur.Imap.Lo_20090101_010101_2009.000_60.00.txt")),
        Path(get_test_instrument_team_data_path("glows/probSur.Imap.Hi_2009.000_135.0.txt")),
        Path(get_test_instrument_team_data_path("glows/probSur.Imap.Hi_2009.000_90.00.txt")),
        Path(get_test_instrument_team_data_path("glows/probSur.Imap.Ul_20090101_010101_2009.000.txt")),
    ]

    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3e',
        start_date=datetime(2015, 4, 10),
        end_date=datetime(2015, 4, 11),
        version='v001',
        descriptor='survival-probability-lo')

    mock_get_repoint_date_range.return_value = (
        np.datetime64(datetime.fromisoformat("2025-04-20T00:00:00")),
        np.datetime64(datetime.fromisoformat("2025-04-21T00:00:00")))

    glows_processor: GlowsProcessor = GlowsProcessor(mock_processing_input_collection, input_metadata)

    glows_processor.process()


@environment_variables({"REPOINT_DATA_FILEPATH": get_test_data_path("fake_1_day_repointing_file.csv")})
@patch("imap_l3_processing.glows.glows_processor.imap_data_access.upload")
def run_glows_l3e_lo_with_less_mocks(_):
    l3d_file = "imap_glows_l3d_solar-hist_20250501-repoint05599_v002.cdf"
    lo_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_energy-grid-lo_20100101_v001.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v001.dat"),
        AncillaryInput("imap_glows_lya-series_20100101_v001.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_20100101_v001.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3e_20100101_v002.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_20100101_v001.dat"),
        AncillaryInput("imap_glows_speed-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_20100101_v001.dat"),
        AncillaryInput("imap_glows_tess-xyz-8_20100101_v001.dat"),
        ScienceInput(l3d_file)
    )

    hi_45_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_energy-grid-hi_20100101_v001.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v001.dat"),
        AncillaryInput("imap_glows_lya-series_20100101_v001.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_20100101_v001.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3e_20100101_v002.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_20100101_v001.dat"),
        AncillaryInput("imap_glows_speed-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_20100101_v001.dat"),
        ScienceInput(l3d_file)
    )
    hi_90_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_energy-grid-hi_20100101_v001.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v001.dat"),
        AncillaryInput("imap_glows_lya-series_20100101_v001.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_20100101_v001.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3e_20100101_v002.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_20100101_v001.dat"),
        AncillaryInput("imap_glows_speed-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_20100101_v001.dat"),
        ScienceInput(l3d_file)
    )

    ul_processing_input_collection = ProcessingInputCollection(
        AncillaryInput("imap_glows_density-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_energy-grid-ultra_20100101_v001.dat"),
        AncillaryInput("imap_glows_ionization-files_20100101_v001.dat"),
        AncillaryInput("imap_glows_lya-series_20100101_v001.dat"),
        AncillaryInput("imap_glows_phion-hydrogen_20100101_v001.dat"),
        AncillaryInput("imap_glows_pipeline-settings-l3e_20100101_v002.json"),
        AncillaryInput("imap_glows_solar-uv-anisotropy_20100101_v001.dat"),
        AncillaryInput("imap_glows_speed-3d_20100101_v001.dat"),
        AncillaryInput("imap_glows_sw-eqtr-electrons_20100101_v001.dat"),
        AncillaryInput("imap_glows_tess-ang-16_20100101_v001.dat"),
        ScienceInput(l3d_file)
    )

    version = 'v007'
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 5, 2)

    lo_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                      end_date=end_date, version=version,
                                      descriptor='survival-probability-lo')

    glows_processor: GlowsProcessor = GlowsProcessor(lo_processing_input_collection, lo_input_metadata)
    glows_processor.process()

    hi_45_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                         end_date=end_date, version=version,
                                         descriptor='survival-probability-hi-45')

    glows_processor: GlowsProcessor = GlowsProcessor(hi_45_processing_input_collection, hi_45_input_metadata)
    glows_processor.process()

    hi_90_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                         end_date=end_date, version=version,
                                         descriptor='survival-probability-hi-90')

    glows_processor: GlowsProcessor = GlowsProcessor(hi_90_processing_input_collection, hi_90_input_metadata)
    glows_processor.process()

    ul_input_metadata = InputMetadata(instrument='glows', data_level='l3e', start_date=start_date,
                                      end_date=end_date, version=version,
                                      descriptor='survival-probability-ul')

    glows_processor: GlowsProcessor = GlowsProcessor(ul_processing_input_collection, ul_input_metadata)
    glows_processor.process()


def run_glows_l3bc():
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3b',
        start_date=datetime(2013, 9, 8),
        end_date=datetime(2013, 9, 8),
        version='v001')

    cr = 2091
    external_files = {
        'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
        'omni_raw_data': get_test_instrument_team_data_path('glows/omni2_all_years.dat')
    }
    ancillary_files = {
        'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
        'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
        'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
        'pipeline_settings': get_test_instrument_team_data_path(
            'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
    }
    l3a_data_folder_path = get_test_data_path('glows/l3a_products')
    l3a_data = []
    l3a_file_names = [f"imap_glows_l3a_hist_201004{x}_v001.cdf" for x in range(22, 31)]
    l3a_file_names += [f"imap_glows_l3a_hist_201005{x:02d}_v001.cdf" for x in range(1, 32)]
    l3a_file_names += [f"imap_glows_l3a_hist_201006{x:02d}_v001.cdf" for x in range(1, 17)]
    for name in l3a_file_names:
        l3a_data.append(create_glows_l3a_dictionary_from_cdf(l3a_data_folder_path / name))

    dependencies = GlowsL3BCDependencies(l3a_data=l3a_data, external_files=external_files,
                                         ancillary_files=ancillary_files, carrington_rotation_number=cr,
                                         start_date=datetime(2009, 12, 20), end_date=datetime(2009, 12, 21),
                                         zip_file_path=Path("fake/path/to/file.zip"))

    processor = GlowsProcessor(Mock(), input_metadata)

    l3b_data_product, l3c_data_product = processor.process_l3bc(dependencies)

    l3b_cdf = save_data(l3b_data_product, delete_if_present=True)
    print(l3b_cdf)

    l3c_data_product.parent_file_names.append(Path(l3b_cdf).name)
    print(save_data(l3c_data_product, delete_if_present=True))


@patch('imap_l3_processing.glows.glows_processor.shutil')
def run_glows_l3d(mock_shutil):
    mock_shutil.move = shutil.copy

    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3d',
        start_date=datetime(2013, 9, 8),
        end_date=datetime(2013, 9, 8),
        version='v001')

    external_files = {
        'lya_raw_data': get_test_data_path('glows/lyman_alpha_composite.nc'),
    }

    ancillary_files = {
        'WawHelioIon': {
            'speed': get_test_data_path('glows/imap_glows_plasma-speed-Legendre-2010a_v001.dat'),
            'p-dens': get_test_data_path('glows/imap_glows_proton-density-Legendre-2010a_v001.dat'),
            'uv-anis': get_test_data_path('glows/imap_glows_uv-anisotropy-2010a_v001.dat'),
            'phion': get_test_data_path('glows/imap_glows_photoion-2010a_v001.dat'),
            'lya': get_test_data_path('glows/imap_glows_lya-2010a_v001.dat'),
            'e-dens': get_test_data_path('glows/imap_glows_electron-density-2010a_v001.dat'),
        },
        'pipeline_settings': get_test_instrument_team_data_path(
            'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
    }

    l3b_file_paths = [
        get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100422_v011.cdf'),
        get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100519_v011.cdf')
    ]

    l3c_file_paths = [
        get_test_data_path('glows/imap_glows_l3c_sw-profile_20100422_v011.cdf'),
        get_test_data_path('glows/imap_glows_l3c_sw-profile_20100519_v011.cdf')
    ]

    l3d_dependencies: GlowsL3DDependencies = GlowsL3DDependencies(external_files=external_files,
                                                                  ancillary_files=ancillary_files,
                                                                  l3b_file_paths=l3b_file_paths,
                                                                  l3c_file_paths=l3c_file_paths)

    processor = GlowsProcessor(ProcessingInputCollection(), input_metadata)
    processor.process_l3d(l3d_dependencies)


def create_empty_hi_l1c_dataset(epoch: datetime, exposures: Optional[np.ndarray] = None,
                                spin_angles: Optional[np.ndarray] = None,
                                energies: Optional[np.ndarray] = None):
    energies = energies if energies is not None else np.geomspace(1, 10000, 9)
    spin_angles = spin_angles if spin_angles is not None else np.arange(0, 360, 0.1) + 0.05
    exposures = exposures if exposures is not None else np.ones(shape=(1, len(energies), len(spin_angles)))

    return xr.Dataset({
        "exposure_times": (
            [
                "epoch",
                "esa_energy_step",
                "hi_pset_spin_angle_bin"
            ],
            exposures
        ),
    },
        coords={
            "epoch": [epoch],
            "esa_energy_step": energies,
            "hi_pset_spin_angle_bin": spin_angles,
        }
    )


def create_empty_glows_l3e_dataset(epoch: datetime, survival_probabilities: np.ndarray,
                                   spin_angles: Optional[np.ndarray] = None,
                                   energies: Optional[np.ndarray] = None):
    energies = energies or np.geomspace(1, 10000, 16)
    spin_angles = spin_angles or np.arange(0, 360, 1) + 0.5

    return xr.Dataset({
        "probability_of_survival": (
            [
                "epoch",
                "energy",
                "spin_angle_bin"
            ],
            survival_probabilities
        )
    },
        coords={
            "epoch": [epoch],
            "energy": energies,
            "spin_angle_bin": spin_angles,
        })


EPOCH = TypeVar("EPOCH")
ENERGY = TypeVar("ENERGY")
LONGITUDE = TypeVar("LONGITUDE")
LATITUDE = TypeVar("LATITUDE")


def read_glows_survival_probability_data_from_cdf() -> tuple[np.ndarray, np.ndarray]:
    l3e = CDF(str(get_test_data_path("glows/imap_glows_l3e_survival-probabilities-hi_20250324_v001.cdf")))
    return l3e["probability_of_survival"][...][:, 0], l3e["probability_of_survival"][...][:, 1]


def create_hi_l3_survival_corrected_cdf(survival_dependencies: HiL3SurvivalDependencies, spacing_degree: int) -> str:
    input_metadata = InputMetadata(instrument="hi",
                                   data_level="l3",
                                   start_date=datetime(2025, 4, 9),
                                   end_date=datetime(2025, 4, 10),
                                   version="v001",
                                   descriptor="h90-ena-h-sf-sp-ram-hae-4deg-6mo",
                                   )

    processor = HiProcessor(Mock(), input_metadata)
    output_data = processor.process_survival_probabilities(survival_dependencies)

    data_product = HiL3IntensityDataProduct(data=output_data, input_metadata=input_metadata)
    return save_data(data_product, delete_if_present=True)


def create_combined_sensor_cdf(combined_dependencies: HiL3CombinedMapDependencies) -> str:
    input_metadata = InputMetadata(
        instrument="hi",
        data_level="l3",
        start_date=datetime(2025, 4, 9),
        end_date=datetime(2025, 4, 10),
        version="v001",
        descriptor="hic-ena-h-hf-nsp-full-hae-4deg-1yr"
    )
    combined_map = combine_maps(combined_dependencies.maps)

    data_product = HiL3IntensityDataProduct(data=combined_map, input_metadata=input_metadata)
    return save_data(data_product, delete_if_present=True)


if __name__ == "__main__":
    if "codice-lo" in sys.argv:
        if "l3a" in sys.argv:
            if "partial-densities" in sys.argv:
                print(create_codice_lo_l3a_partial_densities_cdf())
            elif "direct-events" in sys.argv:
                print(create_codice_lo_l3a_direct_events_cdf())
            elif "3d-instrument-frame" in sys.argv:
                pass
    if "swapi" in sys.argv:
        if "l3a" in sys.argv:
            paths = create_swapi_l3a_cdf(
                "tests/test_data/swapi/imap_swapi_density-temperature-lut_20240905_v000.dat",
                "tests/test_data/swapi/imap_swapi_alpha-density-temperature-lut_20240920_v000.dat",
                "tests/test_data/swapi/imap_swapi_clock-angle-and-flow-deflection-lut_20240918_v000.dat",
                "tests/test_data/swapi/imap_swapi_energy-gf-lut_20240923_v000.dat",
                "tests/test_data/swapi/imap_swapi_instrument-response-lut_20241023_v000.zip",
                "tests/test_data/swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf",
                "tests/test_data/swapi/imap_swapi_l2_50-sweeps_20250606_v001.cdf"
            )
            print(paths)
        if "l3b" in sys.argv:
            path = create_swapi_l3b_cdf(
                "tests/test_data/swapi/imap_swapi_energy-gf-lut_20240923_v000.dat",
                "tests/test_data/swapi/imap_swapi_efficiency-lut_20241020_v000.dat",
                "tests/test_data/swapi/imap_swapi_l2_sci_20100101_v001.cdf")
            print(path)
    if "glows" in sys.argv:
        if "pre-b" in sys.argv:
            run_l3b_initializer()
        elif "l3bc" in sys.argv:
            run_glows_l3bc()
        elif "init-l3bc" in sys.argv:
            run_glows_l3bc_processor_and_initializer()
        elif "l3d" in sys.argv:
            run_glows_l3d()
        elif "l3e" in sys.argv:
            if "mock" in sys.argv:
                run_glows_l3e_lo_with_mocks()
            else:
                run_glows_l3e_lo_with_less_mocks()
        else:
            cdf_data = CDF("tests/test_data/glows/imap_glows_l2_hist_20130908-repoint00001_v004.cdf")
            l2_glows_data = read_l2_glows_data(cdf_data)

            dependencies = GlowsL3ADependencies(l2_glows_data, 5, {
                "calibration_data": Path(
                    "instrument_team_data/glows/imap_glows_calibration-data_20250707_v000.dat"),
                "settings": Path(
                    "instrument_team_data/glows/imap_glows_pipeline-settings_20250707_v002.json"),
                "time_dependent_bckgrd": Path(
                    "instrument_team_data/glows/imap_glows_time-dep-bckgrd_20250707_v000.dat"),
                "extra_heliospheric_bckgrd": Path(
                    "instrument_team_data/glows/imap_glows_map-of-extra-helio-bckgrd_20250707_v000.dat"),
            })

            path = create_glows_l3a_cdf(dependencies)
            print(path)

    if "hit" in sys.argv:
        if "direct_event" in sys.argv:
            path = create_hit_direct_event_cdf()
            print(f"hit direct event data product: {path}")
        else:
            mag_data = read_l1d_mag_data(get_test_data_path("mag/imap_mag_l1d_norm-mago_20250101_v001.cdf"))
            hit_data = read_l2_hit_data(
                get_test_data_path("hit/imap_hit_l2_macropixel-intensity_20250101_v002.cdf"))
            dependencies = HITL3SectoredDependencies(mag_l1d_data=mag_data, data=hit_data)
            print(f"hit macropixel data product: {create_hit_sectored_cdf(dependencies)}")

    if "swe" in sys.argv:
        dependencies = SweL3Dependencies.from_file_paths(
            get_test_data_path("swe/imap_swe_l2_sci_20250630_v002.cdf"),
            get_test_data_path("swe/imap_swe_l1b_sci_20250630_v003.cdf"),
            get_test_data_path("swe/imap_mag_l1d_norm-mago_20250630_v001.cdf"),
            get_test_data_path("swe/imap_swapi_l3a_proton-sw_20250630_v001.cdf"),
            get_test_data_path("swe/example_swe_config.json"),
        )
        print(create_swe_product(dependencies))

    if "swe-fake-spice" in sys.argv:
        dependencies = SweL3Dependencies.from_file_paths(
            get_test_data_path("swe/imap_swe_l2_sci_20250630_v002.cdf"),
            get_test_data_path("swe/imap_swe_l1b_sci_20250630_v003.cdf"),
            get_test_data_path("swe/imap_mag_l1d_norm-mago_20250630_v001.cdf"),
            get_test_data_path("swe/imap_swapi_l3a_proton-sw_20250630_v001.cdf"),
            get_test_data_path("swe/example_swe_config.json"),
        )
        print(create_swe_product_with_fake_spice(dependencies))

    if "hi" in sys.argv:
        hi_targets = ["survival-probability", "spectral-index", "full-spin", "combined-sensors"]
        do_all = not np.any([t in sys.argv for t in hi_targets])

        glows_l3e_folder = get_test_data_path("hi/fake_l3e_survival_probabilities/90")
        glows_l3_paths = list(glows_l3e_folder.iterdir())

        missing_paths, run_local_paths = try_get_many_run_local_paths([
            "hi/full_spin_deps/l1c",
            "hi/full_spin_deps/imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v001.cdf",
            "hi/full_spin_deps/imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20250415_v001.cdf",
            "hi/full_spin_deps/imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v001.cdf",
            "hi/full_spin_deps/imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-6mo_20250415_v001.cdf"
        ])

        if missing_paths:
            create_hi_full_spin_deps(sensor="90")
            create_hi_full_spin_deps(sensor="45")

        [hi_l1c_folder, *map_paths] = run_local_paths
        [l2_ram_90_map_path, l2_antiram_90_map_path,
         l2_ram_45_map_path, l2_antiram_45_map_path] = map_paths
        hi_l1c_paths = list(hi_l1c_folder.iterdir())

        if do_all or "survival-probability" in sys.argv:
            survival_dependencies = HiL3SurvivalDependencies.from_file_paths(
                map_file_path=l2_ram_90_map_path,
                hi_l1c_paths=hi_l1c_paths,
                glows_l3e_paths=glows_l3_paths,
                l2_descriptor="h90-ena-h-sf-nsp-ram-hae-4deg-6mo")
            print(create_hi_l3_survival_corrected_cdf(survival_dependencies, spacing_degree=4))

        if do_all or "spectral-index" in sys.argv:
            dependencies = HiL3SpectralFitDependencies.from_file_paths(
                get_test_data_path("hi/fake_l2_maps/hi45-zirnstein-mondel-6months.cdf")
            )
            print(create_spectral_index_cdf(dependencies))

        if do_all or "full-spin" in sys.argv:
            ram_survival_dependencies = HiL3SurvivalDependencies.from_file_paths(
                map_file_path=l2_ram_90_map_path,
                hi_l1c_paths=hi_l1c_paths,
                glows_l3e_paths=glows_l3_paths,
                l2_descriptor="h90-ena-h-sf-nsp-ram-hae-4deg-6mo")

            antiram_survival_dependencies = HiL3SurvivalDependencies.from_file_paths(
                map_file_path=l2_antiram_90_map_path,
                hi_l1c_paths=hi_l1c_paths,
                glows_l3e_paths=glows_l3_paths,
                l2_descriptor="h90-ena-h-sf-nsp-anti-hae-4deg-6mo")

            full_spin_dependencies = HiL3SingleSensorFullSpinDependencies(
                ram_dependencies=ram_survival_dependencies,
                antiram_dependencies=antiram_survival_dependencies
            )

            print(create_survival_corrected_full_spin_cdf(full_spin_dependencies))
        if do_all or "combined-sensors" in sys.argv:

            missing_paths, run_local_paths = try_get_many_run_local_paths([
                "hi/full_spin_deps/hi90-6months.cdf",
                "hi/full_spin_deps/hi45-6months.cdf",
                "hi/full_spin_deps/imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-4deg-1yr_20250415_v001.cdf",
                "hi/full_spin_deps/imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-4deg-1yr_20250415_v001.cdf",
                "hi/full_spin_deps/imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-4deg-1yr_20250415_v001.cdf",
                "hi/full_spin_deps/imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-4deg-1yr_20250415_v001.cdf"
            ])
            hi90_path, hi45_path, ram90_path, anti90_path, ram45_path, anti45_path = run_local_paths
            if missing_paths:
                shutil.copyfile(hi90_path, ram90_path)
                shutil.copyfile(hi90_path, anti90_path)
                shutil.copyfile(hi45_path, ram45_path)
                shutil.copyfile(hi45_path, anti45_path)

            combined_dependencies = HiL3CombinedMapDependencies.from_file_paths(
                [
                    ram90_path,
                    anti90_path,
                    ram45_path,
                    anti45_path,
                ])
            print(create_combined_sensor_cdf(combined_dependencies))

    if "ultra" in sys.argv:
        if "survival" in sys.argv:

            missing_paths, [l1c_dependency_path, l2_map_path] = try_get_many_run_local_paths([
                "ultra/fake_l1c_psets/test_pset.cdf",
                "ultra/fake_l2_maps/test_l2_map.cdf"
            ])

            if missing_paths:
                _write_ultra_l1c_cdf_with_parents()
                _write_ultra_l2_cdf_with_parents()

            l1c_dependency = UltraL1CPSet.read_from_path(l1c_dependency_path)

            l3e_glows_paths = [
                get_test_data_path(
                    "ultra/fake_l3e_survival_probabilities/imap_glows_l3e_survival-probabilities-ultra_20250901_v001.cdf")
            ]
            l3e_dependencies = [UltraGlowsL3eData.read_from_path(path) for path in l3e_glows_paths]
            l2_map_dependency = UltraL2Map.read_from_path(l2_map_path)

            processor_input_metadata = InputMetadata(
                instrument="ultra",
                start_date=datetime(year=2025, month=9, day=1),
                end_date=datetime(year=2025, month=9, day=1),
                data_level="l3",
                version="v001",
                descriptor="u90-ena-h-sf-sp-full-hae-128nside-6mo"
            )

            dependencies = UltraL3Dependencies(ultra_l1c_pset=[l1c_dependency], glows_l3e_sp=l3e_dependencies,
                                               ultra_l2_map=l2_map_dependency)

            processor = UltraProcessor(input_metadata=processor_input_metadata, dependencies=None)
            output = processor._process_survival_probability(deps=dependencies)

            print(save_data(output, True))
