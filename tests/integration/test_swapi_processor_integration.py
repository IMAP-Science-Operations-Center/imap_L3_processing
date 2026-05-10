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
import numpy.testing
from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF
import datetime

import imap_l3_processing


class SwapiProcessorIntegration(unittest.TestCase):
    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_swapi_processor_with_production_data(self):
        """
        With real data and with full dependency setup, validate that the CDF output
        matches hardcoded expected values to check for unexpected changes
        """

        expected_values = {
            'epoch': datetime.datetime(2025, 12, 31, 23, 59, 35, 6000),
            'proton_sw_speed': 474.57455,
            'proton_sw_speed_uncert': 0.36071813,
            'proton_sw_speed_sun': 475.3688,
            'proton_sw_speed_sun_uncert': 0.3539945,
            'epoch_delta': 30000000000,
            'proton_sw_temperature': 55131.13,
            'proton_sw_temperature_uncert': 2008.7198,
            'proton_sw_density': 2.6919234,
            'proton_sw_density_uncert': 0.049627114,
            'proton_sw_clock_angle': 237.82544,
            'proton_sw_clock_angle_uncert': 1.9230984,
            'proton_sw_deflection_angle': 4.661801,
            'proton_sw_deflection_angle_uncert': 0.10569086,
            'proton_sw_bulk_velocity_rtn_sun': [474.0841, 30.623661, 16.79102],
            'proton_sw_bulk_velocity_rtn_sun_covariance': [[0.1169681, -0.0341877, 0.13108876],
                                                           [-0.0341877, 0.8606747, -0.23923519],
                                                           [0.13108876, -0.23923519, 1.3221142]],
            'proton_sw_bulk_velocity_rtn_sc': [474.14252, 1.0705476, 20.217045],
            'proton_sw_bulk_velocity_rtn_sc_covariance': [[0.1169681, -0.0341877, 0.13108876],
                                                             [-0.0341877, 0.8606747, -0.23923519],
                                                             [0.13108876, -0.23923519, 1.3221142]],
            'swp_flags': 0
        }

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

        # Tolerances are picked above the FP-noise floor of the LM proton fit
        # (numba `fastmath` reordering moves uncertainties by up to ~3% and
        # near-zero perpendicular velocity components by ~0.1 km/s) and below
        # any physics-meaningful regression. Uncertainty fields (including
        # covariance) get a looser rtol because they're amplified by the
        # Jacobian-inverse conditioning. Bulk-velocity components additionally
        # get an absolute floor at 1% of the bulk speed, since the
        # perpendicular components can be sub-km/s where rtol is meaningless.
        UNCERTAINTY_FIELDS = {
            'proton_sw_speed_uncert',
            'proton_sw_speed_sun_uncert',
            'proton_sw_temperature_uncert',
            'proton_sw_density_uncert',
            'proton_sw_clock_angle_uncert',
            'proton_sw_deflection_angle_uncert',
            'proton_sw_bulk_velocity_rtn_sun_covariance',
            'proton_sw_bulk_velocity_rtn_sc_covariance',
        }
        VELOCITY_VECTOR_FIELDS = {
            'proton_sw_bulk_velocity_rtn_sun',
            'proton_sw_bulk_velocity_rtn_sc',
        }
        bulk_speed_atol = 1e-3 * float(expected_values['proton_sw_speed'])

        with CDF(str(expected_file_path)) as cdf:
            for key in expected_values.keys():
                actual_value = cdf[key][0]
                if key in VELOCITY_VECTOR_FIELDS:
                    rtol, atol = 1e-3, bulk_speed_atol
                else:
                    rtol, atol = 1e-3, 0.0
                try:
                    numpy.testing.assert_allclose(
                        actual_value, expected_values[key], rtol=rtol, atol=atol, err_msg=key
                    )
                except TypeError:
                    self.assertEqual(expected_values[key], actual_value, msg=key)



if __name__ == "__main__":
    unittest.main()
