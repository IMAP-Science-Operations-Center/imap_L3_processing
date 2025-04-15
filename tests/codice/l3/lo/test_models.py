import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.models import CodiceLoL2Data


class TestModels(unittest.TestCase):
    def test_lo_l2_sectored_intensities_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            rng = np.random.default_rng()
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                epoch = np.array([datetime(2010, 1, 1), datetime(2010, 1, 2)])
                epoch_delta = np.repeat(len(epoch), 2)
                energy = np.geomspace(2, 1000)
                spin_sector = np.linspace(0, 360, 24)
                ssd_id = np.linspace(0, 360, 16)
                h_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                he_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                c4_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                c5_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                c6_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o5_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o6_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o7_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o8_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                mg_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                si_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                fe_low_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                fe_high_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))

                cdf_file['epoch'] = epoch
                cdf_file['epoch_delta'] = epoch_delta
                cdf_file['energy'] = energy
                cdf_file['spin_sector'] = spin_sector
                cdf_file['ssd_id'] = ssd_id
                cdf_file['h_intensities'] = h_intensities
                cdf_file['he_intensities'] = he_intensities
                cdf_file['c4_intensities'] = c4_intensities
                cdf_file['c5_intensities'] = c5_intensities
                cdf_file['c6_intensities'] = c6_intensities
                cdf_file['o5_intensities'] = o5_intensities
                cdf_file['o6_intensities'] = o6_intensities
                cdf_file['o7_intensities'] = o7_intensities
                cdf_file['o8_intensities'] = o8_intensities
                cdf_file['mg_intensities'] = mg_intensities
                cdf_file['si_intensities'] = si_intensities
                cdf_file['fe_low_intensities'] = fe_low_intensities
                cdf_file['fe_high_intensities'] = fe_high_intensities

            result: CodiceLoL2Data = CodiceLoL2Data.read_from_cdf(cdf_file_path)
            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.epoch_delta, epoch_delta)
            np.testing.assert_array_equal(result.energy, energy)
            np.testing.assert_array_equal(result.spin_sector, spin_sector)
            np.testing.assert_array_equal(result.ssd_id, ssd_id)
            np.testing.assert_array_equal(result.h_intensities, h_intensities)
            np.testing.assert_array_equal(result.he_intensities, he_intensities)
            np.testing.assert_array_equal(result.he_intensities, he_intensities)
            np.testing.assert_array_equal(result.c4_intensities, c4_intensities)
            np.testing.assert_array_equal(result.c5_intensities, c5_intensities)
            np.testing.assert_array_equal(result.c6_intensities, c6_intensities)
            np.testing.assert_array_equal(result.o5_intensities, o5_intensities)
            np.testing.assert_array_equal(result.o6_intensities, o6_intensities)
            np.testing.assert_array_equal(result.o7_intensities, o7_intensities)
            np.testing.assert_array_equal(result.o8_intensities, o8_intensities)
            np.testing.assert_array_equal(result.mg_intensities, mg_intensities)
            np.testing.assert_array_equal(result.si_intensities, si_intensities)
            np.testing.assert_array_equal(result.fe_low_intensities, fe_low_intensities)
