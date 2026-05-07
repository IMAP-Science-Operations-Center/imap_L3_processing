from dataclasses import replace
from datetime import datetime, timedelta, date
from unittest import TestCase
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
from imap_data_access import config
from imap_data_access.processing_input import (
    ProcessingInputCollection,
    ScienceInput,
    AncillaryInput,
)
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing.constants import (
    THIRTY_SECONDS_IN_NANOSECONDS,
    FIVE_MINUTES_IN_NANOSECONDS,
)
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.descriptors import (
    DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR,
    INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR,
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR,
    ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR,
    GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR,
    GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR,
)
from imap_l3_processing.swapi.l3a.models import (
    SwapiL2Data,
    SwapiL3ProtonSolarWindData,
    SwapiL3PickupIonData,
    SwapiL3AlphaSolarWindData,
)
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import FittingParameters
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import (
    SWAPI_L2_DESCRIPTOR,
    SwapiL3ADependencies,
)
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import DeltaMinusPlus
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.uncertainties import (
    make_correlated_velocity,
)
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor


# ---- shared fixtures -------------------------------------------------------

_DEFAULT_COUNT_RATE = np.array(
    [
        [4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23],
    ]
)
_DEFAULT_UNC = np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (4, 1))


def _make_chunk_of_five(initial_epoch: int = 10):
    epoch = np.array([initial_epoch + i for i in range(4)])
    energy = np.tile([15000, 16000, 17000, 18000, 19000], (4, 1))
    return SwapiL2Data(epoch, energy, _DEFAULT_COUNT_RATE.copy(), _DEFAULT_UNC.copy())


def _make_chunk_of_fifty(initial_epoch: int = 10):
    epoch = initial_epoch + np.arange(50)
    energy = np.tile([15000, 16000, 17000, 18000, 19000], (50, 1)) * 2
    return SwapiL2Data(
        epoch,
        energy,
        np.tile(_DEFAULT_COUNT_RATE[0] * 2, (50, 1)).astype(float),
        np.tile(_DEFAULT_UNC[0] * 2, (50, 1)),
    )


def _l3a_input_filenames(
    instrument, dependency_start_date, version, *, geometric_descriptor
):
    return [
        f"imap_{instrument}_l2_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf",
        f"imap_{instrument}_{ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf",
        f"imap_{instrument}_{geometric_descriptor}_{dependency_start_date}_{version}.cdf",
        f"imap_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf",
        f"imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_{version}.cdf",
    ]


def _processing_inputs(file_names):
    science_input = ScienceInput(file_names[0])
    ancillary_inputs = [AncillaryInput(fn) for fn in file_names[1:]]
    return ProcessingInputCollection(science_input, *ancillary_inputs)


def _make_proton_fit_result(
    *,
    bulk_velocity_rtn=np.array([400.0, 10.0, 5.0]),
    density=5.0,
    temperature=12000.0,
    bad_fit_flag=SwapiL3Flags.NONE,
):
    from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
        ProtonSolarWindFitResult,
    )

    return ProtonSolarWindFitResult(
        density=ufloat(density, 0.5),
        temperature=ufloat(temperature, 100.0),
        bulk_velocity_rtn=make_correlated_velocity(bulk_velocity_rtn, np.eye(3)),
        bad_fit_flag=bad_fit_flag,
    )


def _identity_velocity_angles(bulk_velocity_rtn):
    """Return the (speed, clock, deflection) ufloat tuple a mocked
    derive_velocity_angles should produce when its rotation acts as identity."""
    vr, vt, vn = bulk_velocity_rtn
    speed = float(np.linalg.norm(bulk_velocity_rtn))
    clock = float(np.degrees(np.arctan2(vt, vr)) % 360)
    defl = float(np.degrees(np.arccos(-vn / speed)))
    return (ufloat(speed, 0.0), ufloat(clock, 0.0), ufloat(defl, 0.0))


def _make_alpha_moments(bad_fit_flag=SwapiL3Flags.NONE):
    from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.calculate_alpha_solar_wind_moments import (
        AlphaSolarWindMoments,
    )

    return AlphaSolarWindMoments(
        density=ufloat(0.05, 0.01),
        temperature=ufloat(1000.0, 50.0),
        bulk_velocity_rtn=make_correlated_velocity(
            np.array([450.0, 0.0, 0.0]), np.zeros((3, 3))
        ),
        delta_v=ufloat(10.0, 1.0),
        bad_fit_flag=bad_fit_flag,
    )


def create_swapi_l3a_dependencies_with_mocks():
    data = Mock()
    # Default ε_p=ε_α=ε_p_lab so central_effective_area_scale comes out to 1.0 —
    # matches behavior tests written before efficiency wiring.
    efficiency_calibration_table = Mock()
    efficiency_calibration_table.get_proton_efficiency_for.return_value = 0.11
    efficiency_calibration_table.get_alpha_efficiency_for.return_value = 0.11
    efficiency_calibration_table.eps_p_lab = 0.11
    return SwapiL3ADependencies(
        data=data,
        efficiency_calibration_table=efficiency_calibration_table,
        geometric_factor_calibration_table=Mock(),
        instrument_response_calibration_table=Mock(),
        density_of_neutral_helium_calibration_table=Mock(),
        hydrogen_inflow_vector=Mock(),
        helium_inflow_vector=Mock(),
        swapi_response=Mock(),
    )


class TestSwapiProcessor(TestCase):
    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3PickupIonData")
    @patch("imap_l3_processing.utils.write_cdf")
    @patch("imap_l3_processing.swapi.swapi_processor.chunk_l2_data")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.derive_velocity_angles")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_pickup_ion_values")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_ten_minute_velocities")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_density")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_temperature")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry")
    @patch("imap_l3_processing.processor.spiceypy")
    def test_process_l3a_pui(
        self,
        mock_spicepy,
        mock_get_swapi_geometry,
        mock_calculate_helium_pui_temperature,
        mock_calculate_helium_pui_density,
        mock_calculate_ten_minute_velocities,
        mock_calculate_pickup_ion,
        mock_swapi_l3_dependencies_class,
        mock_fit_solar_wind_proton_model,
        mock_derive_velocity_angles,
        mock_chunk_l2_data,
        mock_write_cdf,
        mock_pickup_ion_data_constructor,
        mock_imap_attribute_manager,
    ):
        instrument = "swapi"
        dependency_start_date = "20250101"
        end_date = datetime(2025, 9, 26)
        start_date = datetime(2025, 9, 25)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0
        mock_get_swapi_geometry.return_value = np.tile(np.eye(3), (100, 1, 1))

        bulk_velocity_rtn = np.array([400.0, 10.0, 5.0])
        mock_fit_solar_wind_proton_model.return_value = _make_proton_fit_result(
            bulk_velocity_rtn=bulk_velocity_rtn,
        )
        mock_derive_velocity_angles.return_value = _identity_velocity_angles(
            bulk_velocity_rtn
        )

        chunk_of_five = _make_chunk_of_five()
        chunk_of_fifty = _make_chunk_of_fifty()

        expected_fitting_params = FittingParameters(1, 2, 3, 4)
        mock_calculate_pickup_ion.return_value = expected_fitting_params
        mock_calculate_helium_pui_density.return_value = 5
        mock_calculate_helium_pui_temperature.return_value = 6
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[17, 18, 19]]),
            np.array([SwapiL3Flags.NONE]),
        )

        input_file_names = _l3a_input_filenames(
            instrument,
            dependency_start_date,
            "v001",
            geometric_descriptor=GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR,
        )
        dependencies = _processing_inputs(input_file_names)

        input_metadata = InputMetadata(
            instrument, "l3a", start_date, end_date, input_version
        )
        pickup_ion_data = mock_pickup_ion_data_constructor.return_value
        expected_pickup_ion_metadata = replace(input_metadata, descriptor="pui-he")
        pickup_ion_data.input_metadata = expected_pickup_ion_metadata
        input_metadata.descriptor = "pui-he"

        expected_cdf_path = (
            config["DATA_DIR"]
            / "imap"
            / "swapi"
            / "l3a"
            / "2025"
            / "09"
            / f"imap_swapi_l3a_pui-he_{start_date_as_str}_{input_version}.cdf"
        )

        mock_chunk_l2_data.side_effect = [[chunk_of_five], [chunk_of_fifty]]
        mock_l3a_dependencies = (
            mock_swapi_l3_dependencies_class.fetch_dependencies.return_value
        )
        mock_l3a_dependencies.data = chunk_of_five

        mock_manager = mock_imap_attribute_manager.return_value
        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(
            dependencies
        )
        mock_chunk_l2_data.assert_has_calls(
            [call(chunk_of_five, 5), call(chunk_of_five, 50)]
        )

        # Validate the inputs to calculate_pickup_ion (energies, count_rates,
        # epoch, sw_velocity, calibration tables).
        (
            instrument_response_lut,
            geometric_factor_lut,
            energies,
            count_rates,
            pui_epoch,
            sw_velocity_vector,
            density_of_neutral_helium_lut,
            efficiency_lut,
            hydrogen_inflow_vector,
            helium_inflow_vector,
        ) = mock_calculate_pickup_ion.call_args.args
        self.assertIs(
            mock_l3a_dependencies.instrument_response_calibration_table,
            instrument_response_lut,
        )
        self.assertIs(
            mock_l3a_dependencies.efficiency_calibration_table, efficiency_lut
        )
        self.assertIs(
            mock_l3a_dependencies.geometric_factor_calibration_table,
            geometric_factor_lut,
        )
        np.testing.assert_array_equal(chunk_of_fifty.energy, energies)
        np.testing.assert_array_equal(
            chunk_of_fifty.coincidence_count_rate, count_rates
        )
        self.assertEqual(
            chunk_of_fifty.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS, pui_epoch
        )
        np.testing.assert_array_equal([17, 18, 19], sw_velocity_vector)

        # Ten-minute velocities receive identity-rotation speed/clock/deflection.
        expected_speed = np.linalg.norm(bulk_velocity_rtn)
        vr, vt, vn = bulk_velocity_rtn
        expected_clock = np.degrees(np.arctan2(vt, vr)) % 360
        expected_defl = np.degrees(np.arccos(-vn / expected_speed))
        ten_min_speeds, ten_min_deflections, ten_min_clocks, ten_min_flags = (
            mock_calculate_ten_minute_velocities.call_args.args
        )
        np.testing.assert_allclose(ten_min_speeds, [expected_speed])
        np.testing.assert_allclose(ten_min_deflections, [expected_defl])
        np.testing.assert_allclose(ten_min_clocks, [expected_clock])
        np.testing.assert_array_equal(ten_min_flags, np.array([SwapiL3Flags.NONE]))

        mock_manager.add_global_attribute.assert_has_calls(
            [
                call("Data_version", outgoing_version),
                call("Generation_date", date.today().strftime("%Y%m%d")),
                call("Logical_source", "imap_swapi_l3a_pui-he"),
                call(
                    "Logical_file_id",
                    f"imap_swapi_l3a_pui-he_{start_date_as_str}_{input_version}",
                ),
            ]
        )

        (
            actual_pui_metadata,
            actual_pui_epoch,
            actual_pui_cooling_index,
            actual_pui_ionization_rate,
            actual_pui_cutoff_speed,
            actual_pui_background_rate,
            actual_pui_density,
            actual_pui_temperature,
            actual_quality_flags,
        ) = mock_pickup_ion_data_constructor.call_args.args
        self.assertEqual(expected_pickup_ion_metadata, actual_pui_metadata)
        np.testing.assert_array_equal(
            np.array([10 + FIVE_MINUTES_IN_NANOSECONDS]), actual_pui_epoch
        )
        np.testing.assert_array_equal(np.array([1]), actual_pui_cooling_index)
        np.testing.assert_array_equal(np.array([2]), actual_pui_ionization_rate)
        np.testing.assert_array_equal(np.array([3]), actual_pui_cutoff_speed)
        np.testing.assert_array_equal(np.array([4]), actual_pui_background_rate)
        np.testing.assert_array_equal(np.array([5]), actual_pui_density)
        np.testing.assert_array_equal(np.array([6]), actual_pui_temperature)
        self.assertEqual([0], actual_quality_flags)

        self.assertEqual(input_file_names, pickup_ion_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(
            str(expected_cdf_path), pickup_ion_data, mock_manager
        )
        self.assertEqual([expected_cdf_path], product)

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3ProtonSolarWindData")
    @patch("imap_l3_processing.utils.write_cdf")
    @patch("imap_l3_processing.swapi.swapi_processor.chunk_l2_data")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.derive_velocity_angles")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.get_spacecraft_velocity_rtn")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry")
    @patch("imap_l3_processing.processor.spiceypy")
    def test_process_l3a_proton(
        self,
        mock_spicepy,
        mock_get_swapi_geometry,
        mock_get_spacecraft_velocity_rtn,
        mock_swapi_l3_dependencies_class,
        mock_fit_solar_wind_proton_model,
        mock_derive_velocity_angles,
        mock_chunk_l2_data,
        mock_write_cdf,
        mock_proton_solar_wind_data_constructor,
        mock_imap_attribute_manager,
    ):
        instrument = "swapi"
        dependency_start_date = "20250101"
        end_date = datetime(2025, 6, 13)
        start_date = datetime(2025, 6, 12)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0
        mock_get_swapi_geometry.return_value = np.tile(np.eye(3), (100, 1, 1))
        sc_velocity_rtn = np.array([10.0, -3.0, 2.0])
        mock_get_spacecraft_velocity_rtn.return_value = sc_velocity_rtn

        bulk_velocity_rtn = np.array([400.0, 10.0, 5.0])
        density = 5.0
        temperature = 12000.0
        mock_fit_solar_wind_proton_model.return_value = _make_proton_fit_result(
            bulk_velocity_rtn=bulk_velocity_rtn,
            density=density,
            temperature=temperature,
        )
        mock_derive_velocity_angles.return_value = _identity_velocity_angles(
            bulk_velocity_rtn
        )

        chunk_of_five = _make_chunk_of_five()
        input_file_names = _l3a_input_filenames(
            instrument,
            dependency_start_date,
            "v001",
            geometric_descriptor=GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR,
        )
        dependencies = _processing_inputs(input_file_names)

        input_metadata = InputMetadata(
            instrument, "l3a", start_date, end_date, input_version
        )
        proton_solar_wind_data = mock_proton_solar_wind_data_constructor.return_value
        expected_proton_metadata = replace(input_metadata, descriptor="proton-sw")
        proton_solar_wind_data.input_metadata = expected_proton_metadata
        input_metadata.descriptor = "proton-sw"

        expected_cdf_path = (
            config["DATA_DIR"]
            / "imap"
            / "swapi"
            / "l3a"
            / "2025"
            / "06"
            / f"imap_swapi_l3a_proton-sw_{start_date_as_str}_{input_version}.cdf"
        )

        mock_chunk_l2_data.side_effect = [[chunk_of_five]]
        swapi_l3a_dependencies = create_swapi_l3a_dependencies_with_mocks()
        swapi_l3a_dependencies = replace(swapi_l3a_dependencies, data=chunk_of_five)
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = (
            swapi_l3a_dependencies
        )

        mock_manager = mock_imap_attribute_manager.return_value
        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(
            dependencies
        )
        mock_chunk_l2_data.assert_has_calls([call(chunk_of_five, 5)])

        (actual_proton_metadata,) = (
            mock_proton_solar_wind_data_constructor.call_args.args
        )
        kwargs = mock_proton_solar_wind_data_constructor.call_args.kwargs

        self.assertEqual(expected_proton_metadata, actual_proton_metadata)
        np.testing.assert_array_equal(
            np.array([10 + THIRTY_SECONDS_IN_NANOSECONDS]),
            kwargs["epoch"],
            strict=True,
        )

        expected_speed = np.linalg.norm(bulk_velocity_rtn)
        expected_sun_speed = np.linalg.norm(bulk_velocity_rtn + sc_velocity_rtn)
        self.assertAlmostEqual(kwargs["proton_sw_speed"][0], expected_speed)
        self.assertAlmostEqual(kwargs["proton_sw_speed_sun"][0], expected_sun_speed)
        self.assertAlmostEqual(kwargs["proton_sw_temperature"][0], temperature)
        self.assertAlmostEqual(kwargs["proton_sw_density"][0], density)

        vr, vt, vn = bulk_velocity_rtn
        expected_clock = np.degrees(np.arctan2(vt, vr)) % 360
        expected_defl = np.degrees(np.arccos(-vn / expected_speed))
        self.assertAlmostEqual(kwargs["proton_sw_clock_angle"][0], expected_clock)
        self.assertAlmostEqual(kwargs["proton_sw_deflection_angle"][0], expected_defl)

        np.testing.assert_array_equal(
            np.array([SwapiL3Flags.NONE]), kwargs["quality_flags"], strict=True
        )
        np.testing.assert_allclose(
            kwargs["proton_sw_bulk_velocity_rtn_sun"][0],
            bulk_velocity_rtn + sc_velocity_rtn,
        )
        np.testing.assert_allclose(
            kwargs["proton_sw_bulk_velocity_rtn_sc"][0], bulk_velocity_rtn
        )

        mock_manager.add_global_attribute.assert_has_calls(
            [
                call("Data_version", outgoing_version),
                call("Generation_date", date.today().strftime("%Y%m%d")),
                call("Logical_source", "imap_swapi_l3a_proton-sw"),
                call(
                    "Logical_file_id",
                    f"imap_swapi_l3a_proton-sw_{start_date_as_str}_{input_version}",
                ),
            ]
        )

        self.assertEqual(input_file_names, proton_solar_wind_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(
            str(expected_cdf_path), proton_solar_wind_data, mock_manager
        )
        self.assertEqual([expected_cdf_path], product)

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.write_cdf")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3ProtonSolarWindData")
    @patch("imap_l3_processing.swapi.swapi_processor.chunk_l2_data")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.derive_velocity_angles")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.get_spacecraft_velocity_rtn")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry")
    @patch("imap_l3_processing.processor.spiceypy")
    def test_process_l3a_proton_propagates_bad_fit_flag(
        self,
        mock_spicepy,
        mock_get_swapi_geometry,
        mock_get_spacecraft_velocity_rtn,
        mock_swapi_l3_dependencies_class,
        mock_fit_solar_wind_proton_model,
        mock_derive_velocity_angles,
        mock_chunk_l2_data,
        mock_proton_solar_wind_data_constructor,
        _,
        __,
    ):
        instrument = "swapi"
        dependency_start_date = "20250101"
        end_date = datetime(2025, 6, 13)
        start_date = datetime(2025, 6, 12)

        mock_spicepy.ktotal.return_value = 0
        mock_get_swapi_geometry.return_value = np.tile(np.eye(3), (100, 1, 1))
        mock_get_spacecraft_velocity_rtn.return_value = np.zeros(3)

        mock_fit_solar_wind_proton_model.return_value = _make_proton_fit_result(
            bulk_velocity_rtn=np.array([400.0, 0.0, 0.0]),
            density=5.0,
            temperature=10000.0,
            bad_fit_flag=SwapiL3Flags.FIT_FAILED,
        )
        mock_derive_velocity_angles.return_value = (
            ufloat(400.0, 0.0),
            ufloat(0.0, 0.0),
            ufloat(90.0, 0.0),
        )

        chunk_of_five = _make_chunk_of_five()
        input_file_names = _l3a_input_filenames(
            instrument,
            dependency_start_date,
            "v001",
            geometric_descriptor=GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR,
        )
        dependencies = _processing_inputs(input_file_names)
        input_metadata = InputMetadata(instrument, "l3a", start_date, end_date, "v123")
        input_metadata.descriptor = "proton-sw"
        proton_solar_wind_data = mock_proton_solar_wind_data_constructor.return_value
        proton_solar_wind_data.input_metadata = replace(
            input_metadata, descriptor="proton-sw"
        )

        mock_chunk_l2_data.side_effect = [[chunk_of_five]]
        swapi_l3a_dependencies = create_swapi_l3a_dependencies_with_mocks()
        swapi_l3a_dependencies = replace(swapi_l3a_dependencies, data=chunk_of_five)
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = (
            swapi_l3a_dependencies
        )

        SwapiProcessor(dependencies, input_metadata).process()

        actual_quality_flags = mock_proton_solar_wind_data_constructor.call_args.kwargs[
            "quality_flags"
        ]
        np.testing.assert_array_equal(
            np.array([SwapiL3Flags.FIT_FAILED]), actual_quality_flags, strict=True
        )

    def test_process_l3a_proton_outputs_fill_for_chunks_with_fill(self):
        # SPICE is unavailable in this test, so precompute_geometry hits the
        # EPHEMERIS_GAP path before extract_coarse_sweep ever sees the NaN bin.
        epoch = np.array([10, 11, 12, 13, 14])
        energy = np.tile([15000, 16000, 17000, 18000, 19000], (5, 1))
        coincidence_count_rate = np.array(
            [
                [4, 5, 6, 7, 8],
                [9, 10, 11, np.nan, 13],
                [14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28],
            ]
        )
        chunk = SwapiL2Data(
            epoch, energy, coincidence_count_rate, np.tile(_DEFAULT_UNC[0], (5, 1))
        )
        input_metadata = InputMetadata(
            "swapi", "l3a", datetime(2025, 6, 12), datetime(2025, 6, 13), "v123"
        )
        swapi_processor = SwapiProcessor(Mock(), input_metadata)
        product = swapi_processor.process_l3a_proton(data=chunk, dependencies=Mock())

        self.assertIsInstance(product, SwapiL3ProtonSolarWindData)
        for field in (
            "proton_sw_speed",
            "proton_sw_speed_uncert",
            "proton_sw_temperature",
            "proton_sw_temperature_uncert",
            "proton_sw_density",
            "proton_sw_density_uncert",
            "proton_sw_clock_angle",
            "proton_sw_clock_angle_uncert",
            "proton_sw_deflection_angle",
            "proton_sw_deflection_angle_uncert",
        ):
            np.testing.assert_array_equal(getattr(product, field), [np.nan])
        np.testing.assert_array_equal(
            product.quality_flags, [int(SwapiL3Flags.EPHEMERIS_GAP)]
        )

    def test_process_l3a_alpha_outputs_fill_for_chunks_with_fill(self):
        epoch = np.array([10, 11, 12, 13, 14])
        energy = np.tile([19000, 18000, 17000, 16000, 15000], (5, 1))
        coincidence_count_rate = np.array(
            [
                [4, 5, 6, 7, 8],
                [9, 10, 11, np.nan, 13],
                [14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28],
            ]
        )
        chunk = SwapiL2Data(
            epoch, energy, coincidence_count_rate, np.tile(_DEFAULT_UNC[0], (5, 1))
        )
        input_metadata = InputMetadata(
            "swapi", "l3a", datetime(2025, 6, 12), datetime(2025, 6, 13), "v123"
        )
        swapi_processor = SwapiProcessor(Mock(), input_metadata)
        product = swapi_processor.process_l3a_alpha(data=chunk, dependencies=Mock())

        self.assertIsInstance(product, SwapiL3AlphaSolarWindData)
        np.testing.assert_array_equal(product.alpha_sw_density, [np.nan])
        np.testing.assert_array_equal(product.alpha_sw_temperature, [np.nan])
        np.testing.assert_array_equal(product.alpha_sw_delta_v, [np.nan])

    def test_process_l3a_pui_outputs_fill_for_chunks_with_fill(self):
        # 50 sweeps so chunk_l2_data(data, 50) yields one outer chunk; otherwise
        # ten-minute velocities and bad_fit_flags end up with mismatched lengths
        # and the final bitwise_or fails.
        n_sweeps = 50
        epoch = 10 + np.arange(n_sweeps)
        energy = np.tile([15000, 16000, 17000, 18000, 19000], (n_sweeps, 1))
        coincidence_count_rate = np.tile([4.0, 5.0, 6.0, 7.0, 8.0], (n_sweeps, 1))
        coincidence_count_rate[1, 3] = np.nan
        chunk = SwapiL2Data(
            epoch,
            energy,
            coincidence_count_rate,
            np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (n_sweeps, 1)),
        )
        input_metadata = InputMetadata(
            "swapi", "l3a", datetime(2025, 6, 12), datetime(2025, 6, 13), "v123"
        )
        swapi_processor = SwapiProcessor(Mock(), input_metadata)
        product = swapi_processor.process_l3a_pui(data=chunk, dependencies=Mock())

        self.assertIsInstance(product, SwapiL3PickupIonData)
        np.testing.assert_array_equal(product.epoch, 10 + FIVE_MINUTES_IN_NANOSECONDS)
        for field in (
            "cooling_index",
            "ionization_rate",
            "cutoff_speed",
            "background_rate",
            "density",
            "temperature",
        ):
            arr = getattr(product, field)
            np.testing.assert_array_equal(nominal_values(arr), [np.nan])
            np.testing.assert_array_equal(std_devs(arr), [np.nan])
        # SPICE is unavailable, so PuiProtonChunkFitter raises EPHEMERIS_GAP
        # which propagates through calculate_ten_minute_velocities.
        np.testing.assert_array_equal(
            product.quality_flags, [int(SwapiL3Flags.EPHEMERIS_GAP)]
        )

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3AlphaSolarWindData")
    @patch("imap_l3_processing.utils.write_cdf")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.fit_solar_wind_alpha_moments")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton")
    @patch("imap_l3_processing.swapi.swapi_processor.chunk_l2_data")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies")
    @patch("imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry")
    @patch("imap_l3_processing.processor.spiceypy")
    def test_process_l3a_alpha(
        self,
        mock_spicepy,
        mock_get_swapi_geometry,
        mock_swapi_l3_dependencies_class,
        mock_chunk_l2_data,
        mock_fit_solar_wind_proton_model,
        mock_fit_solar_wind_alpha_moments,
        mock_write_cdf,
        mock_alpha_solar_wind_data_constructor,
        mock_imap_attribute_manager,
    ):
        instrument = "swapi"
        dependency_start_date = "20250101"
        end_date = datetime(2025, 8, 29)
        start_date = datetime(2025, 8, 28)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0
        mock_get_swapi_geometry.return_value = np.tile(np.eye(3), (100, 1, 1))

        chunk_of_five = SwapiL2Data(
            np.array([10, 11, 12, 13]),
            np.tile([15000, 16000, 17000, 18000, 19000], (4, 1)),
            _DEFAULT_COUNT_RATE.copy(),
            np.zeros_like(_DEFAULT_COUNT_RATE, dtype=float),
        )

        # alpha-sw uses the SW geometric factor descriptor (no PUI / α-T-N descriptor).
        input_file_names = [
            f"imap_{instrument}_l2_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_v001.cdf",
            f"imap_{instrument}_{GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_v001.cdf",
            f"imap_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_v001.cdf",
            f"imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_v001.cdf",
        ]
        dependencies = _processing_inputs(input_file_names)

        input_metadata = InputMetadata(
            instrument, "l3a", start_date, end_date, input_version
        )

        mock_fit_solar_wind_proton_model.return_value = _make_proton_fit_result(
            bulk_velocity_rtn=np.array([440.0, 0.0, 0.0]),
            density=5.0,
            temperature=10000.0,
        )
        # AlphaSolarWindMoments must be a real (picklable) dataclass, not a Mock,
        # because it is returned from a forked worker process via the result queue.
        alpha_moments = _make_alpha_moments()
        mock_fit_solar_wind_alpha_moments.return_value = alpha_moments

        alpha_solar_wind_data = mock_alpha_solar_wind_data_constructor.return_value
        expected_alpha_metadata = replace(input_metadata, descriptor="alpha-sw")
        alpha_solar_wind_data.input_metadata = expected_alpha_metadata
        input_metadata.descriptor = "alpha-sw"

        expected_cdf_path = (
            config["DATA_DIR"]
            / "imap"
            / "swapi"
            / "l3a"
            / "2025"
            / "08"
            / f"imap_swapi_l3a_alpha-sw_{start_date_as_str}_{input_version}.cdf"
        )

        mock_chunk_l2_data.side_effect = [[chunk_of_five]]

        # epoch=11 falls inside [10, 10+2*THIRTY_SECONDS_IN_NANOSECONDS], giving
        # AlphaChunkFitter.precompute_geometry a finite b_hat.
        mock_mag_data = Mock()
        mock_mag_data.epoch = np.array([11])
        mock_mag_data.mag_data = np.array([[1.0, 0.0, 0.0]])
        swapi_l3a_dependencies = create_swapi_l3a_dependencies_with_mocks()
        swapi_l3a_dependencies = replace(
            swapi_l3a_dependencies, data=chunk_of_five, mag_data=mock_mag_data
        )
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = (
            swapi_l3a_dependencies
        )

        mock_manager = mock_imap_attribute_manager.return_value
        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(
            dependencies
        )
        mock_chunk_l2_data.assert_has_calls([call(chunk_of_five, 5)])

        mock_manager.add_global_attribute.assert_has_calls(
            [
                call("Data_version", outgoing_version),
                call("Generation_date", date.today().strftime("%Y%m%d")),
                call("Logical_source", "imap_swapi_l3a_alpha-sw"),
                call(
                    "Logical_file_id",
                    f"imap_swapi_l3a_alpha-sw_{start_date_as_str}_{input_version}",
                ),
            ]
        )

        call_args = mock_alpha_solar_wind_data_constructor.call_args
        self.assertEqual(expected_alpha_metadata, call_args.args[0])
        np.testing.assert_array_equal(
            np.array([10 + THIRTY_SECONDS_IN_NANOSECONDS]),
            call_args.kwargs["epoch"],
        )
        np.testing.assert_array_equal(
            call_args.kwargs["alpha_sw_density"],
            np.array([alpha_moments.density.nominal_value]),
        )
        np.testing.assert_array_equal(
            call_args.kwargs["bad_fit_flag"], np.array([int(SwapiL3Flags.NONE)])
        )

        self.assertEqual(input_file_names, alpha_solar_wind_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(
            str(expected_cdf_path), alpha_solar_wind_data, mock_manager
        )
        self.assertEqual([expected_cdf_path], product)

    def _alpha_preliminary_mag_setup(self, mag_is_preliminary):
        """Run process_l3a_alpha with mag_is_preliminary set; return product."""
        with (
            patch(
                "imap_l3_processing.swapi.l3a.chunk_fits.fit_solar_wind_alpha_moments"
            ) as mock_alpha,
            patch("imap_l3_processing.swapi.l3a.chunk_fits._fit_proton") as mock_proton,
            patch(
                "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry"
            ) as mock_geom,
        ):
            mock_geom.return_value = np.tile(np.eye(3), (100, 1, 1))
            mock_proton.return_value = _make_proton_fit_result(
                bulk_velocity_rtn=np.array([440.0, 0.0, 0.0]),
                density=5.0,
                temperature=10000.0,
            )
            mock_alpha.return_value = _make_alpha_moments()

            epoch = np.array([10, 11, 12, 13, 14])
            energy = np.tile([15000, 16000, 17000, 18000, 19000], (5, 1))
            coincidence_count_rate = np.array(
                [
                    [4, 5, 6, 7, 8],
                    [9, 10, 11, 12, 13],
                    [14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23],
                    [24, 25, 26, 27, 28],
                ],
                dtype=float,
            )
            chunk = SwapiL2Data(
                epoch,
                energy,
                coincidence_count_rate,
                np.zeros_like(coincidence_count_rate),
            )

            mock_mag_data = Mock()
            mock_mag_data.epoch = np.array([11])
            mock_mag_data.mag_data = np.array([[1.0, 0.0, 0.0]])

            dependencies = Mock()
            dependencies.data = chunk
            dependencies.mag_data = mock_mag_data
            dependencies.mag_is_preliminary = mag_is_preliminary

            input_metadata = InputMetadata(
                "swapi", "l3a", datetime(2025, 6, 12), datetime(2025, 6, 13), "v123"
            )
            return SwapiProcessor(Mock(), input_metadata).process_l3a_alpha(
                chunk, dependencies
            )

    def test_process_l3a_alpha_sets_preliminary_mag_flag_when_mag_is_l1d(self):
        product = self._alpha_preliminary_mag_setup(mag_is_preliminary=True)
        self.assertTrue(
            np.all(np.asarray(product.bad_fit_flag) & int(SwapiL3Flags.PRELIMINARY_MAG))
        )

    def test_process_l3a_alpha_does_not_set_preliminary_mag_flag_when_mag_is_l2(self):
        product = self._alpha_preliminary_mag_setup(mag_is_preliminary=False)
        self.assertFalse(
            np.any(np.asarray(product.bad_fit_flag) & int(SwapiL3Flags.PRELIMINARY_MAG))
        )

    @patch("imap_l3_processing.swapi.swapi_processor.calculate_delta_minus_plus")
    @patch("imap_l3_processing.swapi.swapi_processor.save_data")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3BCombinedVDF")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_combined_sweeps")
    @patch("imap_l3_processing.swapi.swapi_processor.chunk_l2_data")
    @patch("imap_l3_processing.swapi.swapi_processor.SwapiL3BDependencies")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_alpha_solar_wind_vdf")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_proton_solar_wind_vdf")
    @patch("imap_l3_processing.swapi.swapi_processor.calculate_pui_solar_wind_vdf")
    @patch(
        "imap_l3_processing.swapi.swapi_processor.calculate_combined_solar_wind_differential_flux"
    )
    @patch("imap_l3_processing.processor.spiceypy")
    def test_process_l3b(
        self,
        mock_spiceypy,
        mock_calculate_combined_solar_wind_differential_flux,
        mock_calculate_pui_solar_wind_vdf,
        mock_calculate_proton_solar_wind_vdf,
        mock_calculate_alpha_solar_wind_vdf,
        mock_swapi_l3b_dependencies_class,
        mock_chunk_l2_data,
        mock_calculate_combined_sweeps,
        mock_combined_vdf_data,
        mock_save_data,
        mock_calculate_delta_minus_plus,
    ):
        instrument = "swapi"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        outgoing_version = "12345"
        dependency_start_date = "20250101"

        mock_spiceypy.ktotal.return_value = 0
        mock_calculate_proton_solar_wind_vdf.side_effect = [
            (
                sentinel.proton_calculated_velocities1,
                sentinel.proton_calculated_probabilities1,
            ),
            (
                sentinel.proton_calculated_velocities2,
                sentinel.proton_calculated_probabilities2,
            ),
        ]
        mock_calculate_alpha_solar_wind_vdf.side_effect = [
            (
                sentinel.alpha_calculated_velocities1,
                sentinel.alpha_calculated_probabilities1,
            ),
            (
                sentinel.alpha_calculated_velocities2,
                sentinel.alpha_calculated_probabilities2,
            ),
        ]
        mock_calculate_pui_solar_wind_vdf.side_effect = [
            (
                sentinel.pui_calculated_velocities1,
                sentinel.pui_calculated_probabilities1,
            ),
            (
                sentinel.pui_calculated_velocities2,
                sentinel.pui_calculated_probabilities2,
            ),
        ]
        mock_calculate_combined_solar_wind_differential_flux.side_effect = [
            sentinel.calculated_diffential_flux1,
            sentinel.calculated_diffential_flux2,
        ]
        mock_calculate_delta_minus_plus.side_effect = [
            DeltaMinusPlus(
                sentinel.proton_velocity_delta_minus1,
                sentinel.proton_velocity_delta_plus1,
            ),
            DeltaMinusPlus(
                sentinel.alpha_velocity_delta_minus1,
                sentinel.alpha_velocity_delta_plus1,
            ),
            DeltaMinusPlus(
                sentinel.pui_velocity_delta_minus1, sentinel.pui_velocity_delta_plus1
            ),
            DeltaMinusPlus(sentinel.energy_delta_minus1, sentinel.energy_delta_plus1),
            DeltaMinusPlus(
                sentinel.proton_velocity_delta_minus2,
                sentinel.proton_velocity_delta_plus2,
            ),
            DeltaMinusPlus(
                sentinel.alpha_velocity_delta_minus2,
                sentinel.alpha_velocity_delta_plus2,
            ),
            DeltaMinusPlus(
                sentinel.pui_velocity_delta_minus2, sentinel.pui_velocity_delta_plus2
            ),
            DeltaMinusPlus(sentinel.energy_delta_minus2, sentinel.energy_delta_plus2),
        ]

        energy = np.array([15000, 16000, 17000, 18000, 19000])
        average_count_rates = [14, 15, 16, 17, 18]
        average_count_rate_uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5]
        coincidence_count_rate = _DEFAULT_COUNT_RATE.copy()
        coincidence_count_rate_uncertainty = _DEFAULT_UNC.copy()

        first_initial_epoch = 10
        first_chunk = SwapiL2Data(
            np.array([first_initial_epoch, 11, 12, 13]),
            sentinel.energies,
            coincidence_count_rate,
            coincidence_count_rate_uncertainty,
        )
        second_initial_epoch = 60
        second_chunk = SwapiL2Data(
            np.array([second_initial_epoch, 11, 12, 13]),
            sentinel.energies,
            coincidence_count_rate,
            coincidence_count_rate_uncertainty,
        )
        mock_chunk_l2_data.return_value = [first_chunk, second_chunk]

        mock_calculate_combined_sweeps.return_value = [
            uarray(average_count_rates, average_count_rate_uncertainties),
            energy,
        ]

        input_file_names = [
            f"imap_{instrument}_l2_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_v001.cdf",
            f"imap_{instrument}_l2_{GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_v001.cdf",
            f"imap_{instrument}_l2_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_v001.cdf",
        ]
        dependencies = ProcessingInputCollection(
            *[ScienceInput(fn) for fn in input_file_names]
        )
        input_metadata = InputMetadata(
            instrument, "l3b", start_date, end_date, outgoing_version
        )

        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3b_dependencies_class.fetch_dependencies.assert_called_once_with(
            dependencies
        )
        deps_return = mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value
        mock_geometric = deps_return.geometric_factor_calibration_table
        mock_efficiency = deps_return.efficiency_calibration_table
        mock_chunk_l2_data.assert_called_with(deps_return.data, 50)

        # combined_sweeps was called with the count-rate uarray and the chunk's energy.
        first_combined_args = mock_calculate_combined_sweeps.call_args_list[0].args
        np.testing.assert_array_equal(
            coincidence_count_rate, nominal_values(first_combined_args[0])
        )
        np.testing.assert_array_equal(
            coincidence_count_rate_uncertainty, std_devs(first_combined_args[0])
        )
        self.assertEqual(sentinel.energies, first_combined_args[1])

        mock_efficiency.get_proton_efficiency_for.assert_has_calls(
            [
                call(first_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS),
                call(second_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS),
            ]
        )

        expected_count_rate = uarray(
            average_count_rates, average_count_rate_uncertainties
        )
        nominal = nominal_values(expected_count_rate)
        sigmas = std_devs(expected_count_rate)

        # Each VDF/diff-flux helper is called with (energy, count_rate, eff, geom).
        for vdf_mock in (
            mock_calculate_proton_solar_wind_vdf,
            mock_calculate_alpha_solar_wind_vdf,
            mock_calculate_pui_solar_wind_vdf,
            mock_calculate_combined_solar_wind_differential_flux,
        ):
            args = vdf_mock.call_args_list[0].args
            np.testing.assert_array_equal(energy, args[0])
            np.testing.assert_array_equal(nominal, nominal_values(args[1]))
            np.testing.assert_array_equal(sigmas, std_devs(args[1]))
            self.assertIs(
                mock_efficiency.get_proton_efficiency_for.return_value, args[2]
            )
            self.assertIs(mock_geometric, args[3])

        mock_calculate_delta_minus_plus.assert_has_calls(
            [
                call(sentinel.proton_calculated_velocities1),
                call(sentinel.alpha_calculated_velocities1),
                call(sentinel.pui_calculated_velocities1),
                call(energy),
                call(sentinel.proton_calculated_velocities2),
                call(sentinel.alpha_calculated_velocities2),
                call(sentinel.pui_calculated_velocities2),
                call(energy),
            ]
        )

        expected_metadata = InputMetadata(
            descriptor="combined",
            data_level="l3b",
            start_date=start_date,
            end_date=end_date,
            instrument=instrument,
            version=outgoing_version,
        )
        kwargs = mock_combined_vdf_data.call_args_list[0].kwargs
        self.assertEqual(expected_metadata, kwargs["input_metadata"])
        np.testing.assert_array_equal(
            np.array(
                [
                    first_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS,
                    second_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS,
                ]
            ),
            kwargs["epoch"],
        )

        # Spot-check that each species' velocities/dv_minus/dv_plus/vdf row are
        # the per-chunk sentinels from the side_effects list above.
        for species, vels, dvm, dvp, vdf in (
            (
                "proton",
                [
                    sentinel.proton_calculated_velocities1,
                    sentinel.proton_calculated_velocities2,
                ],
                [
                    sentinel.proton_velocity_delta_minus1,
                    sentinel.proton_velocity_delta_minus2,
                ],
                [
                    sentinel.proton_velocity_delta_plus1,
                    sentinel.proton_velocity_delta_plus2,
                ],
                [
                    sentinel.proton_calculated_probabilities1,
                    sentinel.proton_calculated_probabilities2,
                ],
            ),
            (
                "alpha",
                [
                    sentinel.alpha_calculated_velocities1,
                    sentinel.alpha_calculated_velocities2,
                ],
                [
                    sentinel.alpha_velocity_delta_minus1,
                    sentinel.alpha_velocity_delta_minus2,
                ],
                [
                    sentinel.alpha_velocity_delta_plus1,
                    sentinel.alpha_velocity_delta_plus2,
                ],
                [
                    sentinel.alpha_calculated_probabilities1,
                    sentinel.alpha_calculated_probabilities2,
                ],
            ),
            (
                "pui",
                [
                    sentinel.pui_calculated_velocities1,
                    sentinel.pui_calculated_velocities2,
                ],
                [
                    sentinel.pui_velocity_delta_minus1,
                    sentinel.pui_velocity_delta_minus2,
                ],
                [sentinel.pui_velocity_delta_plus1, sentinel.pui_velocity_delta_plus2],
                [
                    sentinel.pui_calculated_probabilities1,
                    sentinel.pui_calculated_probabilities2,
                ],
            ),
        ):
            np.testing.assert_array_equal(vels, kwargs[f"{species}_sw_velocities"])
            np.testing.assert_array_equal(
                dvm, kwargs[f"{species}_sw_velocities_delta_minus"]
            )
            np.testing.assert_array_equal(
                dvp, kwargs[f"{species}_sw_velocities_delta_plus"]
            )
            np.testing.assert_array_equal(vdf, kwargs[f"{species}_sw_combined_vdf"])

        np.testing.assert_array_equal([energy, energy], kwargs["combined_energy"])
        np.testing.assert_array_equal(
            [sentinel.energy_delta_minus1, sentinel.energy_delta_minus2],
            kwargs["combined_energy_delta_minus"],
        )
        np.testing.assert_array_equal(
            [sentinel.energy_delta_plus1, sentinel.energy_delta_plus2],
            kwargs["combined_energy_delta_plus"],
        )
        np.testing.assert_array_equal(
            [
                sentinel.calculated_diffential_flux1,
                sentinel.calculated_diffential_flux2,
            ],
            kwargs["combined_differential_flux"],
        )

        self.assertEqual(
            input_file_names, mock_combined_vdf_data.return_value.parent_file_names
        )
        mock_save_data.assert_called_once_with(mock_combined_vdf_data.return_value)
        self.assertEqual([mock_save_data.return_value], product)
