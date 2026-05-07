"""End-to-end subprocess integration test for the SWAPI L3a proton-sw processor.

Mirrors test_swe_processor_integration.py: runs ``imap_l3_data_processor.py`` as
a subprocess against staged test data, exercising the full real path —
dependency manifest deserialization → SPICE furnishing → SwapiL3ADependencies
loading of all 13 ancillaries → SwapiProcessor.process() → process_l3a_proton
→ save_data → CDF written to disk.

SWAPI-specific inputs live in ``tests/integration/test_data/swapi/``. SPICE
kernels are pulled from the shared ``tests/integration/test_data/spice/`` dir.
The L2 science CDF is generated on the fly by retiming an existing synthetic
spectrum into the SPICE coverage window — keeping a date-shifted copy on disk
would just be duplicate data.
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import skipUnless

import numpy as np
import imap_data_access
from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF

import imap_l3_processing


class SwapiProcessorIntegration(unittest.TestCase):
    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_swapi_processor_with_production_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        imap_data_access.config["DATA_DIR"] = root_dir / "data"

        anc_dir = root_dir / "data" / "imap" / "ancillary" / "swapi"
        anc_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "imap_swapi_azimuthal-transmission_20260425_v001.csv",
            "imap_swapi_central-effective-area_20260425_v001.csv",
            "imap_swapi_passband-fit-coefficients_20260425_v001.csv",
        ]:
            dest = anc_dir / name
            if not dest.exists():
                shutil.copy(root_dir / "instrument_team_data" / "swapi" / name, dest)

        expected_file_path = ScienceFilePath(
            "imap_swapi_l3a_proton-sw_20260101_v001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        result = subprocess.run(
            [
                sys.executable,
                "imap_l3_data_processor.py",
                "--instrument",
                "swapi",
                "--data-level",
                "l3a",
                "--descriptor",
                "proton-sw",
                "--start-date",
                "20260101",
                "--version",
                "v001",
                "--dependency",
                # TODO switch to dependency file
                """
                [{"type":"science","files":["imap_swapi_l2_sci_20260101_v001.cdf"]},{"type":"ancillary","files":["imap_swapi_alpha-density-temperature-lut_20250125_v001.dat"]},{"type":"ancillary","files":["imap_swapi_efficiency-lut_20241020_v001.dat"]},{"type":"ancillary","files":["imap_swapi_energy-gf-pui-lut_20100101_v003.csv"]},{"type":"ancillary","files":["imap_swapi_instrument-response-lut_20241023_v001.zip"]},{"type":"ancillary","files":["imap_swapi_density-of-neutral-helium-lut_20241023_v002.dat"]},{"type":"ancillary","files":["imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"]},{"type":"ancillary","files":["imap_swapi_helium-inflow-vector_20100101_v001.dat"]},{"type":"ancillary","files":["imap_swapi_azimuthal-transmission_20260425_v001.csv"]},{"type":"ancillary","files":["imap_swapi_central-effective-area_20260425_v001.csv"]},{"type":"ancillary","files":["imap_swapi_passband-fit-coefficients_20260425_v001.csv"]},{"type":"spice","files":["naif0012.tls","pck00011.tpc","imap_130.tf","imap_science_120.tf","imap_sclk_0161.tsc","de440.bsp","imap_recon_20250925_20260420_v01.bsp","imap_2025_358_2026_085_004.ah.bc","imap_dps_2025_363_2025_365_001.ah.bc","imap_dps_2025_359_2026_115_002.ah.bc"]}]
                """,
                # "imap_swapi_l3a_proton-sw_20260425_v001.json",
            ]
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

        # process_l3a_proton swallows per-chunk exceptions and writes fill values,
        # so returncode==0 alone doesn't prove the fitter ran. Open the CDF and
        # check that at least one chunk produced a finite, physical solar-wind speed.
        with CDF(str(expected_file_path)) as cdf:
            speed_fill = float(cdf["proton_sw_speed"].attrs["FILLVAL"])
            speeds = np.asarray(cdf["proton_sw_speed"][...], dtype=float)
            temperatures = np.asarray(cdf["proton_sw_temperature"][...], dtype=float)
            densities = np.asarray(cdf["proton_sw_density"][...], dtype=float)
            clock_angles = np.asarray(cdf["proton_sw_clock_angle"][...], dtype=float)
            deflection_angles = np.asarray(
                cdf["proton_sw_deflection_angle"][...], dtype=float
            )
            bulk_vel_sun = np.asarray(
                cdf["proton_sw_bulk_velocity_rtn_sun"][...], dtype=float
            )
            bulk_vel_sc = np.asarray(
                cdf["proton_sw_bulk_velocity_rtn_sc"][...], dtype=float
            )
            speed_uncerts = np.asarray(cdf["proton_sw_speed_uncert"][...], dtype=float)
            temp_uncerts = np.asarray(
                cdf["proton_sw_temperature_uncert"][...], dtype=float
            )
            density_uncerts = np.asarray(
                cdf["proton_sw_density_uncert"][...], dtype=float
            )
            clock_angle_uncerts = np.asarray(
                cdf["proton_sw_clock_angle_uncert"][...], dtype=float
            )
            deflection_angle_uncerts = np.asarray(
                cdf["proton_sw_deflection_angle_uncert"][...], dtype=float
            )
            flags = np.asarray(cdf["swp_flags"][...])

        valid = np.isfinite(speeds) & (speeds != speed_fill)
        finite = speeds[valid]
        self.assertGreater(len(finite), 0, "no chunk produced a finite proton_sw_speed")
        self.assertTrue(
            np.all((finite > 200.0) & (finite < 1500.0)),
            f"finite speeds outside plausible heliospheric range: {finite}",
        )

        # Regression check: spot-check 3 time indices against hardcoded values
        # from a known-good run. Tolerances are 1% relative.
        chk = [0, 4, -1]
        np.testing.assert_allclose(
            speeds[chk],
            [474.576, 475.232, 485.351],
            rtol=0.01,
            err_msg="proton_sw_speed regression",
        )
        np.testing.assert_allclose(
            temperatures[chk],
            [55132.4, 65904.9, 274370.7],
            rtol=0.01,
            err_msg="proton_sw_temperature regression",
        )
        np.testing.assert_allclose(
            densities[chk],
            [2.692, 3.254, 4.984],
            rtol=0.01,
            err_msg="proton_sw_density regression",
        )
        np.testing.assert_allclose(
            clock_angles[chk],
            [237.827, 257.576, 294.811],
            rtol=0.01,
            err_msg="proton_sw_clock_angle regression",
        )
        np.testing.assert_allclose(
            deflection_angles[chk],
            [4.661, 5.040, 4.579],
            rtol=0.01,
            err_msg="proton_sw_deflection_angle regression",
        )
        np.testing.assert_allclose(
            bulk_vel_sun[chk],
            [
                [474.086, 30.619, 16.787],
                [475.046, 37.297, 4.385],
                [485.009, 28.077, -19.966],
            ],
            rtol=0.01,
            err_msg="proton_sw_bulk_velocity_rtn_sun regression",
        )
        np.testing.assert_allclose(
            bulk_vel_sc[chk],
            [
                [474.144, 1.066, 20.213],
                [475.105, 7.744, 7.811],
                [485.067, -1.476, -16.540],
            ],
            rtol=0.01,
            err_msg="proton_sw_bulk_velocity_rtn_sc regression",
        )
        np.testing.assert_allclose(
            speed_uncerts[chk],
            [0.178, 0.180, 0.312],
            rtol=0.01,
            err_msg="proton_sw_speed_uncert regression",
        )
        np.testing.assert_allclose(
            temp_uncerts[chk],
            [818.983, 907.037, 2767.607],
            rtol=0.01,
            err_msg="proton_sw_temperature_uncert regression",
        )
        np.testing.assert_allclose(
            density_uncerts[chk],
            [0.023, 0.029, 0.030],
            rtol=0.01,
            err_msg="proton_sw_density_uncert regression",
        )
        # Clock/deflection uncerts are MC-propagated (seed=0); for this real-
        # data window v_xy is well above σ_xy (deflection ~5°), so the angles
        # are well-determined and σ comes out to a few degrees rather than the
        # ~100° saturation seen on spin-aligned synthetic plasma.
        np.testing.assert_allclose(
            clock_angle_uncerts[chk],
            [0.970, 0.834, 1.459],
            rtol=0.01,
            err_msg="proton_sw_clock_angle_uncert regression",
        )
        np.testing.assert_allclose(
            deflection_angle_uncerts[chk],
            [0.066, 0.065, 0.104],
            rtol=0.01,
            err_msg="proton_sw_deflection_angle_uncert regression",
        )
        np.testing.assert_array_equal(
            flags,
            np.zeros(len(flags), dtype=np.uint16),
            err_msg="swp_flags should all be 0",
        )


if __name__ == "__main__":
    unittest.main()
