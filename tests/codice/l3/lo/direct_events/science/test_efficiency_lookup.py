import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from imap_l3_processing.codice.l3.lo.constants import CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_AZIMUTH_BINS
from imap_l3_processing.codice.l3.lo.direct_events.science.efficiency_lookup import EfficiencyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    EventDirection


class TestEfficiencyLookup(unittest.TestCase):
    def test_efficiency_lookup(self):
        num_species = 2
        num_azimuths = 3
        num_energies = 4

        efficiency_lookup = EfficiencyLookup.create_with_fake_data(num_species, num_azimuths, num_energies)

        self.assertEqual((num_species, num_azimuths, num_energies), efficiency_lookup.efficiency_data.shape)

    def test_read_from_zip(self):
        rng = np.random.default_rng()

        mass_species_bin_lookup = MassSpeciesBinLookup(
            _range_to_species={
                "sw_species": ["sw_cno", "sw_hplus"],
                "nsw_species": ["nsw_hplus", "nsw_heplus2"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            sw_cno_efficiency = rng.random((CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_AZIMUTH_BINS))
            sw_hplus_efficiency = rng.random((CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_AZIMUTH_BINS))
            nsw_hplus_efficiency = rng.random((CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_AZIMUTH_BINS))
            nsw_heplus2_efficiency = rng.random((CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_AZIMUTH_BINS))

            sw_cno_efficiency_path = tmpdir / "sw_cno-efficiency.csv"
            sw_hplus_efficiency_path = tmpdir / "sw_hplus-efficiency.csv"
            nsw_hplus_efficiency_path = tmpdir / "nsw_hplus-efficiency.csv"
            nsw_heplus2_efficiency_path = tmpdir / "nsw_heplus2-efficiency.csv"

            np.savetxt(sw_cno_efficiency_path, sw_cno_efficiency, delimiter=",", header="# some header")
            np.savetxt(sw_hplus_efficiency_path, sw_hplus_efficiency, delimiter=",", header="# some header")
            np.savetxt(nsw_hplus_efficiency_path, nsw_hplus_efficiency, delimiter=",", header="# some header")
            np.savetxt(nsw_heplus2_efficiency_path, nsw_heplus2_efficiency, delimiter=",", header="# some header")

            zip_file_path = tmpdir / "test-efficiency-factors.zip"
            with zipfile.ZipFile(zip_file_path, "w") as zip_file:
                for file_path in [
                    sw_cno_efficiency_path,
                    sw_hplus_efficiency_path,
                    nsw_hplus_efficiency_path,
                    nsw_heplus2_efficiency_path
                ]:
                    zip_file.write(file_path, file_path.name)

            efficiency_lookup = EfficiencyLookup.read_from_zip(zip_file_path, mass_species_bin_lookup)

            self.assertEqual((4, 24, 128), efficiency_lookup.efficiency_data.shape)

            sw_cno_index = mass_species_bin_lookup.get_species_index("sw_cno", EventDirection.Sunward)
            np.testing.assert_array_equal(efficiency_lookup.efficiency_data[sw_cno_index], sw_cno_efficiency.T)

            sw_hplus_index = mass_species_bin_lookup.get_species_index("sw_hplus", EventDirection.Sunward)
            np.testing.assert_array_equal(efficiency_lookup.efficiency_data[sw_hplus_index], sw_hplus_efficiency.T)

            nsw_hplus_index = mass_species_bin_lookup.get_species_index("nsw_hplus", EventDirection.NonSunward)
            np.testing.assert_array_equal(efficiency_lookup.efficiency_data[nsw_hplus_index], nsw_hplus_efficiency.T)

            nsw_heplus2_index = mass_species_bin_lookup.get_species_index("nsw_heplus2", EventDirection.NonSunward)
            np.testing.assert_array_equal(efficiency_lookup.efficiency_data[nsw_heplus2_index],
                                          nsw_heplus2_efficiency.T)
