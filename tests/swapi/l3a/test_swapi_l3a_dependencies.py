"""Tests for `SwapiL3ADependencies`.

End-to-end loads against the real test fixtures: `from_file_paths` is called
with actual paths, and `fetch_dependencies` is exercised through the
integration `mock_imap_data_access` helper that stages files into a temp SDC
tree. The returned dependency objects are then *used* (not just inspected) —
calling `lookup_geometric_factor`, `get_proton_efficiency_for`, etc. — so a
regression that returns a structurally-valid-but-non-functional object surfaces
here. A separate class uses mocks to verify the MAG L2/L1D fallback selection
in `fetch_dependencies`, since the integration helper does not stage MAG.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from imap_data_access.processing_input import (
    AncillaryInput,
    ProcessingInputCollection,
    ScienceInput,
)
from spacepy.pycdf import lib as pycdf_lib

from imap_l3_processing.models import MagData
from imap_l3_processing.swapi.descriptors import (
    AZIMUTHAL_TRANSMISSION_DESCRIPTOR,
    CENTRAL_EFFECTIVE_AREA_DESCRIPTOR,
    DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR,
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR,
    GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR,
    HELIUM_INFLOW_VECTOR_DESCRIPTOR,
    HYDROGEN_INFLOW_VECTOR_DESCRIPTOR,
    INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR,
    PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR,
    SWAPI_L2_DESCRIPTOR,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
)
from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import (
    GeometricFactorCalibrationTable,
)
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import (
    InstrumentResponseLookupTableCollection,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path


# Test-fixture filenames that live under tests/test_data/swapi or instrument_team_data/swapi.
_L2_SCIENCE = get_test_data_path("swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf")
_EFFICIENCY = get_test_data_path("swapi/imap_swapi_efficiency-lut_20241020_v000.dat")
_GEOMETRIC_FACTOR_PUI = get_test_data_path(
    "swapi/imap_swapi_energy-gf-pui-lut_20100101_v001.csv"
)
_INSTRUMENT_RESPONSE = get_test_data_path(
    "swapi/imap_swapi_instrument-response-lut_20241023_v000.zip"
)
_NEUTRAL_HELIUM = get_test_data_path(
    "swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf"
)
_HYDROGEN_INFLOW = get_test_data_path(
    "swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"
)
_HELIUM_INFLOW = get_test_data_path(
    "swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"
)
_AZIMUTHAL_TRANSMISSION = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


class TestFromFilePaths(unittest.TestCase):
    """`from_file_paths` is the constructor that maps disk paths to loaded objects.
    Verify it produces a fully usable `SwapiL3ADependencies` from real fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.deps = SwapiL3ADependencies.from_file_paths(
            _L2_SCIENCE,
            _EFFICIENCY,
            _GEOMETRIC_FACTOR_PUI,
            _INSTRUMENT_RESPONSE,
            _NEUTRAL_HELIUM,
            _HYDROGEN_INFLOW,
            _HELIUM_INFLOW,
            _AZIMUTHAL_TRANSMISSION,
            _CENTRAL_EFFECTIVE_AREA,
            _PASSBAND_FIT_COEFFICIENTS,
        )

    def test_l2_data_is_loaded_and_usable(self):
        self.assertIsInstance(self.deps.data, SwapiL2Data)
        # Sample fixture has 50 sweeps × 72 bins.
        self.assertEqual(self.deps.data.energy.shape[1], 72)
        self.assertGreater(len(self.deps.data.sci_start_time), 0)
        self.assertTrue(np.all(np.isfinite(self.deps.data.sci_start_time)))

    def test_efficiency_table_returns_real_value_for_valid_epoch(self):
        self.assertIsInstance(
            self.deps.efficiency_calibration_table, EfficiencyCalibrationTable
        )
        # The fixture's table has entries spanning 2000 → 2024; pick a 2024 epoch.
        epoch = pycdf_lib.datetime_to_tt2000(datetime(2024, 11, 1))
        result = self.deps.efficiency_calibration_table.get_proton_efficiency_for(epoch)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0.0)

    def test_geometric_factor_table_interpolates(self):
        self.assertIsInstance(
            self.deps.geometric_factor_calibration_table,
            GeometricFactorCalibrationTable,
        )
        # Pick an energy in-grid; result is finite and positive.
        result = self.deps.geometric_factor_calibration_table.lookup_geometric_factor(
            np.array([1000.0, 8000.0])
        )
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result > 0))

    def test_instrument_response_collection_resolves_known_bin(self):
        self.assertIsInstance(
            self.deps.instrument_response_calibration_table,
            InstrumentResponseLookupTableCollection,
        )
        table = (
            self.deps.instrument_response_calibration_table.get_table_for_energy_bin(2)
        )
        self.assertGreater(len(table.energy), 0)
        self.assertTrue(np.all(np.isfinite(table.response)))

    def test_neutral_helium_table_returns_finite_density(self):
        self.assertIsInstance(
            self.deps.density_of_neutral_helium_calibration_table,
            DensityOfNeutralHeliumLookupTable,
        )
        # Distance ≥ table minimum must produce finite densities.
        d_min = (
            self.deps.density_of_neutral_helium_calibration_table.get_minimum_distance()
        )
        result = self.deps.density_of_neutral_helium_calibration_table.density(
            np.array([0.0, 90.0]), np.array([d_min, d_min])
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_inflow_vectors_have_expected_components(self):
        self.assertIsInstance(self.deps.hydrogen_inflow_vector, InflowVector)
        self.assertIsInstance(self.deps.helium_inflow_vector, InflowVector)
        # The fixture H+ vector pinned in the inflow_vector tests has speed=22, lat=9.
        self.assertEqual(self.deps.hydrogen_inflow_vector.speed_km_per_s, 22.0)
        # The fixture He vector has speed=25.4.
        self.assertEqual(self.deps.helium_inflow_vector.speed_km_per_s, 25.4)

    def test_swapi_response_is_loaded_and_can_warm_cache(self):
        self.assertIsInstance(self.deps.swapi_response, SwapiResponse)
        # The cache has not been populated yet — warm one voltage and confirm
        # `create_passband_grid` then resolves it.
        v = 1000.0 / 1.89
        self.deps.swapi_response.warm_cache([v])
        grid = self.deps.swapi_response.create_passband_grid(v)
        self.assertGreater(grid.values_open_aperture.size, 0)

    def test_mag_data_default_is_none(self):
        self.assertIsNone(self.deps.mag_data)

    def test_mag_is_preliminary_default_is_false(self):
        self.assertFalse(self.deps.mag_is_preliminary)


class TestFromFilePathsWithMagOptional(unittest.TestCase):
    """`from_file_paths` accepts an optional `mag_path` for the alpha-sw branch.
    When provided it must produce a `MagData` with the documented validmin/max
    sanitization applied; when omitted (default) it stays `None` and
    `mag_is_preliminary` defaults to `False`."""

    def test_mag_path_loads_into_mag_data(self):
        # Build a tiny synthetic MAG CDF, point `from_file_paths` at it, confirm
        # `mag_data` is a `MagData` with the expected shape.
        from spacepy.pycdf import CDF

        with tempfile.TemporaryDirectory() as tmpdir:
            mag_path = Path(tmpdir) / "mag.cdf"
            cdf = CDF(str(mag_path), "")
            cdf["epoch"] = [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)]
            cdf["b_rtn"] = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
            cdf["b_rtn"].attrs["FILLVAL"] = -1e31
            cdf["b_rtn"].attrs["VALIDMIN"] = -1.0e5
            cdf["b_rtn"].attrs["VALIDMAX"] = 1.0e5
            cdf.close()

            deps = SwapiL3ADependencies.from_file_paths(
                _L2_SCIENCE,
                _EFFICIENCY,
                _GEOMETRIC_FACTOR_PUI,
                _INSTRUMENT_RESPONSE,
                _NEUTRAL_HELIUM,
                _HYDROGEN_INFLOW,
                _HELIUM_INFLOW,
                _AZIMUTHAL_TRANSMISSION,
                _CENTRAL_EFFECTIVE_AREA,
                _PASSBAND_FIT_COEFFICIENTS,
                mag_path,
            )
        self.assertIsInstance(deps.mag_data, MagData)
        self.assertEqual(deps.mag_data.mag_data.shape, (2, 3))
        np.testing.assert_array_equal(deps.mag_data.mag_data[0], [1.0, 2.0, 3.0])
        # mag_is_preliminary defaults to False when not passed explicitly.
        self.assertFalse(deps.mag_is_preliminary)


def _staged_input_collection_no_mag() -> ProcessingInputCollection:
    """Build a `ProcessingInputCollection` whose file paths point at the real
    fixtures (re-named to SDC convention so `mock_imap_data_access`'s
    `generate_imap_file_path` parses them)."""
    start_date = "20100105"
    version = "v010"
    science = ScienceInput(
        f"imap_swapi_l2_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf"
    )
    ancillaries = [
        AncillaryInput(f"imap_swapi_{descriptor}_{start_date}_{version}.{ext}")
        for descriptor, ext in [
            (EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, "dat"),
            (GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR, "csv"),
            (INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR, "zip"),
            (DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, "dat"),
            (HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, "dat"),
            (HELIUM_INFLOW_VECTOR_DESCRIPTOR, "dat"),
            (AZIMUTHAL_TRANSMISSION_DESCRIPTOR, "csv"),
            (CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, "csv"),
            (PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR, "csv"),
        ]
    ]
    collection = ProcessingInputCollection()
    collection.add([science, *ancillaries])
    return collection


def _staged_input_files() -> list[Path]:
    """Real source files re-staged with names that match the SDC-style file paths
    in the input collection above. `mock_imap_data_access` copies these into a
    temp SDC tree using `generate_imap_file_path` to derive the destination."""
    start_date = "20100105"
    version = "v010"
    return [
        # The L2 CDF must have a name parseable as a ScienceFilePath; copy under SDC name.
        _stage_under_name(_L2_SCIENCE, f"imap_swapi_l2_sci_{start_date}_{version}.cdf"),
        _stage_under_name(
            _EFFICIENCY, f"imap_swapi_efficiency-lut_{start_date}_{version}.dat"
        ),
        _stage_under_name(
            _GEOMETRIC_FACTOR_PUI,
            f"imap_swapi_energy-gf-pui-lut_{start_date}_{version}.csv",
        ),
        _stage_under_name(
            _INSTRUMENT_RESPONSE,
            f"imap_swapi_instrument-response-lut_{start_date}_{version}.zip",
        ),
        _stage_under_name(
            _NEUTRAL_HELIUM,
            f"imap_swapi_density-of-neutral-helium-lut_{start_date}_{version}.dat",
        ),
        _stage_under_name(
            _HYDROGEN_INFLOW,
            f"imap_swapi_hydrogen-inflow-vector_{start_date}_{version}.dat",
        ),
        _stage_under_name(
            _HELIUM_INFLOW,
            f"imap_swapi_helium-inflow-vector_{start_date}_{version}.dat",
        ),
        _stage_under_name(
            _AZIMUTHAL_TRANSMISSION,
            f"imap_swapi_azimuthal-transmission_{start_date}_{version}.csv",
        ),
        _stage_under_name(
            _CENTRAL_EFFECTIVE_AREA,
            f"imap_swapi_central-effective-area_{start_date}_{version}.csv",
        ),
        _stage_under_name(
            _PASSBAND_FIT_COEFFICIENTS,
            f"imap_swapi_passband-fit-coefficients_{start_date}_{version}.csv",
        ),
    ]


_STAGE_DIR = Path(tempfile.mkdtemp(prefix="swapi_dep_test_staging_"))


def _stage_under_name(source: Path, dest_name: str) -> Path:
    """Copy `source` into the staging dir under `dest_name` so it can be processed
    by `mock_imap_data_access`'s file-path parser."""
    import shutil

    _STAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _STAGE_DIR / dest_name
    if not dest.exists() or dest.stat().st_mtime < source.stat().st_mtime:
        shutil.copy(source, dest)
    return dest


class TestFetchDependenciesEndToEnd(unittest.TestCase):
    """End-to-end exercise of `fetch_dependencies` against a staged SDC data dir.
    Patches `imap_data_access.download` (via the integration helper) so each
    `download(...)` call resolves to a real file on disk.
    """

    def test_fetch_dependencies_returns_usable_dependencies(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with mock_imap_data_access(Path(data_dir), _staged_input_files()):
                collection = _staged_input_collection_no_mag()
                deps = SwapiL3ADependencies.fetch_dependencies(collection)
        self.assertIsInstance(deps.data, SwapiL2Data)
        self.assertIsInstance(
            deps.efficiency_calibration_table, EfficiencyCalibrationTable
        )
        self.assertIsInstance(
            deps.geometric_factor_calibration_table, GeometricFactorCalibrationTable
        )
        self.assertIsInstance(
            deps.instrument_response_calibration_table,
            InstrumentResponseLookupTableCollection,
        )
        self.assertIsInstance(
            deps.density_of_neutral_helium_calibration_table,
            DensityOfNeutralHeliumLookupTable,
        )
        self.assertIsInstance(deps.hydrogen_inflow_vector, InflowVector)
        self.assertIsInstance(deps.helium_inflow_vector, InflowVector)
        self.assertIsInstance(deps.swapi_response, SwapiResponse)
        self.assertIsNone(deps.mag_data)
        self.assertFalse(deps.mag_is_preliminary)

        # The lookup tables must be functional (not just constructed).
        epoch = pycdf_lib.datetime_to_tt2000(datetime(2024, 11, 1))
        self.assertGreater(
            deps.efficiency_calibration_table.get_proton_efficiency_for(epoch), 0
        )
        self.assertGreater(
            deps.geometric_factor_calibration_table.lookup_geometric_factor(
                np.array([1000.0])
            )[0],
            0,
        )


def _input_collection_with_mag(*mag_inputs: ScienceInput) -> ProcessingInputCollection:
    """SWAPI L3A input collection with all required ancillaries plus optional MAG.

    Used only by the mock-based MAG-fallback tests below; real fixture loading
    is exercised by `TestFetchDependenciesEndToEnd`."""
    collection = ProcessingInputCollection()
    collection.add(
        [
            ScienceInput("imap_swapi_l2_sci_20100105_v010.cdf"),
            AncillaryInput(
                f"imap_swapi_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_20100105_v010.cdf"
            ),
            AncillaryInput(
                f"imap_swapi_{GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR}_20100105_v010.cdf"
            ),
            AncillaryInput(
                f"imap_swapi_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_20100105_v010.cdf"
            ),
            AncillaryInput(
                f"imap_swapi_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_20100105_v010.cdf"
            ),
            AncillaryInput(
                f"imap_swapi_{HYDROGEN_INFLOW_VECTOR_DESCRIPTOR}_20100105_v010.cdf"
            ),
            AncillaryInput(
                f"imap_swapi_{HELIUM_INFLOW_VECTOR_DESCRIPTOR}_20100105_v010.cdf"
            ),
            AncillaryInput(
                f"imap_swapi_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_20100105_v010.csv"
            ),
            AncillaryInput(
                f"imap_swapi_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_20100105_v010.csv"
            ),
            AncillaryInput(
                f"imap_swapi_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_20100105_v010.csv"
            ),
            *mag_inputs,
        ]
    )
    return collection


class TestFetchDependenciesMagSelection(unittest.TestCase):
    """`fetch_dependencies` selects between MAG L2 (preferred) and MAG L1D
    (preliminary fallback) based on which descriptors appear in the
    `ProcessingInputCollection`. These tests mock `from_file_paths` and the
    `download` calls so the selection logic is exercised independently of
    real CDF I/O — see `TestFetchDependenciesEndToEnd` for the full path."""

    @patch(
        "imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiL3ADependencies.from_file_paths"
    )
    @patch("imap_l3_processing.utils.download")
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.download")
    def test_uses_mag_l1d_when_only_l1d_present(
        self, mock_download, mock_utils_download, mock_from_file_paths
    ):
        collection = _input_collection_with_mag(
            ScienceInput("imap_mag_l1d_norm-rtn_20100105_v010.cdf"),
        )
        mock_download.side_effect = lambda path: f"downloaded:{path}"
        mock_utils_download.side_effect = lambda path: f"downloaded:{path}"

        SwapiL3ADependencies.fetch_dependencies(collection)

        mag_path_arg = mock_from_file_paths.call_args.args[-2]
        mag_is_preliminary_arg = mock_from_file_paths.call_args.args[-1]
        self.assertIn("mag_l1d_norm-rtn", str(mag_path_arg))
        self.assertTrue(mag_is_preliminary_arg)

    @patch(
        "imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiL3ADependencies.from_file_paths"
    )
    @patch("imap_l3_processing.utils.download")
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.download")
    def test_prefers_mag_l2_when_both_present(
        self, mock_download, mock_utils_download, mock_from_file_paths
    ):
        collection = _input_collection_with_mag(
            ScienceInput("imap_mag_l1d_norm-rtn_20100105_v010.cdf"),
            ScienceInput("imap_mag_l2_norm-rtn_20100105_v010.cdf"),
        )
        mock_download.side_effect = lambda path: f"downloaded:{path}"
        mock_utils_download.side_effect = lambda path: f"downloaded:{path}"

        SwapiL3ADependencies.fetch_dependencies(collection)

        mag_path_arg = mock_from_file_paths.call_args.args[-2]
        mag_is_preliminary_arg = mock_from_file_paths.call_args.args[-1]
        self.assertIn("mag_l2_norm-rtn", str(mag_path_arg))
        self.assertNotIn("mag_l1d", str(mag_path_arg))
        self.assertFalse(mag_is_preliminary_arg)

    @patch(
        "imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiL3ADependencies.from_file_paths"
    )
    @patch("imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.download")
    def test_passes_none_when_no_mag_present(self, mock_download, mock_from_file_paths):
        collection = _input_collection_with_mag()
        mock_download.side_effect = lambda path: f"downloaded:{path}"

        SwapiL3ADependencies.fetch_dependencies(collection)

        self.assertIsNone(mock_from_file_paths.call_args.args[-2])
        self.assertFalse(mock_from_file_paths.call_args.args[-1])


if __name__ == "__main__":
    unittest.main()
