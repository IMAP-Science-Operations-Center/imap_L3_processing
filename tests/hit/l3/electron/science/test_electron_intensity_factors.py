import unittest
from pathlib import Path

import numpy as np

from imap_l3_processing.hit.l3.electron.science.electron_intensity_factors import (
    ElectronIntensityFactors,
)
from tests.test_helpers import get_test_data_path, with_tempdir

FIXTURE_PATH = get_test_data_path(
    "hit/electron/imap_hit_electron-intensity-factors_20260513_v001.csv"
)


class TestElectronIntensityFactors(unittest.TestCase):
    def test_from_file_parses_attached_ancillary_into_per_bin_factors_and_energy_bins(
        self,
    ) -> None:
        factors = ElectronIntensityFactors.from_file(FIXTURE_PATH)

        np.testing.assert_array_equal(factors.a_side_rate_indices, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(factors.b_side_rate_indices, [10, 11, 12, 13, 14, 15])
        np.testing.assert_allclose(factors.delta_e_mev, [0.3, 0.1, 0.4, 0.4, 0.8, 88])
        np.testing.assert_allclose(
            factors.geometry_factor_cm2_sr,
            [1.7792, 0.39942, 0.20431, 0.011787, 0.0013381, 0.000034886],
        )
        np.testing.assert_allclose(factors.background, [0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(factors.energy_lower_mev, [0, 0.3, 0.4, 0.8, 1.2, 2.0])
        np.testing.assert_allclose(factors.energy_upper_mev, [0.3, 0.4, 0.8, 1.2, 2.0, 90.0])
        self.assertEqual(
            factors.energy_labels,
            [
                "<300 keV",
                "300-400 keV",
                "400-800 keV",
                "800-1200 keV",
                "1200-2000 keV",
                ">2000 keV",
            ],
        )

    @with_tempdir
    def test_from_file_rejects_malformed_ancillary(self, temp_dir: Path) -> None:
        header, *rows = FIXTURE_PATH.read_text(encoding="utf-8-sig").splitlines()
        defects = {
            "five data rows": ([header, *rows[:5]], "Expected 6 energy bins"),
            "unexpected header": (["rate_a,rate_b,dE,G,b", *rows], "Unexpected header"),
        }
        for defect, (lines, message_fragment) in defects.items():
            path = temp_dir / f"{defect}.csv"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.subTest(defect=defect):
                with self.assertRaises(ValueError) as context:
                    ElectronIntensityFactors.from_file(path)
                self.assertIn(str(path), str(context.exception))
                self.assertIn(message_fragment, str(context.exception))
