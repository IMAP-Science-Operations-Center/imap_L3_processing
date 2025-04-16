import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, sentinel

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.models import CodiceLoL2Data, CodiceLoL3aDataProduct


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

    def test_get_species(self):
        h_intensities = np.array([sentinel.h_intensities])
        he_intensities = np.array([sentinel.he_intensities])
        c4_intensities = np.array([sentinel.c4_intensities])
        c5_intensities = np.array([sentinel.c5_intensities])
        c6_intensities = np.array([sentinel.c6_intensities])
        o5_intensities = np.array([sentinel.o5_intensities])
        o6_intensities = np.array([sentinel.o6_intensities])
        o7_intensities = np.array([sentinel.o7_intensities])
        o8_intensities = np.array([sentinel.o8_intensities])
        mg_intensities = np.array([sentinel.mg_intensities])
        si_intensities = np.array([sentinel.si_intensities])
        fe_low_intensities = np.array([sentinel.fe_low_intensities])
        fe_high_intensities = np.array([sentinel.fe_high_intensities])

        l2_data_product = CodiceLoL2Data(Mock(), Mock(), Mock(), Mock(), Mock(), h_intensities, he_intensities,
                                         c4_intensities, c5_intensities, c6_intensities, o5_intensities, o6_intensities,
                                         o7_intensities, o8_intensities, mg_intensities, si_intensities,
                                         fe_low_intensities, fe_high_intensities)

        species_intensities = l2_data_product.get_species_intensities()

        np.testing.assert_array_equal(species_intensities['H+'], h_intensities)
        np.testing.assert_array_equal(species_intensities['He++'], he_intensities)
        np.testing.assert_array_equal(species_intensities['C+4'], c4_intensities)
        np.testing.assert_array_equal(species_intensities['C+5'], c5_intensities)
        np.testing.assert_array_equal(species_intensities['C+6'], c6_intensities)
        np.testing.assert_array_equal(species_intensities['O+5'], o5_intensities)
        np.testing.assert_array_equal(species_intensities['O+6'], o6_intensities)
        np.testing.assert_array_equal(species_intensities['O+7'], o7_intensities)
        np.testing.assert_array_equal(species_intensities['O+8'], o8_intensities)
        np.testing.assert_array_equal(species_intensities['Mg'], mg_intensities)
        np.testing.assert_array_equal(species_intensities['Si'], si_intensities)
        np.testing.assert_array_equal(species_intensities['Fe (low Q)'], fe_low_intensities)
        np.testing.assert_array_equal(species_intensities['Fe (high Q)'], fe_high_intensities)

    def test_codice_lo_l3a_to_data_product(self):
        epoch_data = np.array([datetime.now()])

        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([10]),
            "h_partial_density": np.array([15]),
            "he_partial_density": np.array([15]),
            "c4_partial_density": np.array([15]),
            "c5_partial_density": np.array([15]),
            "c6_partial_density": np.array([15]),
            "o5_partial_density": np.array([15]),
            "o6_partial_density": np.array([15]),
            "o7_partial_density": np.array([15]),
            "o8_partial_density": np.array([15]),
            "mg_partial_density": np.array([15]),
            "si_partial_density": np.array([15]),
            "fe_low_partial_density": np.array([15]),
            "fe_high_partial_density": np.array([15]),
        }

        data_product = CodiceLoL3aDataProduct(
            **input_data_product_kwargs
        )
        actual_data_product_variables = data_product.to_data_product_variables()

        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)
