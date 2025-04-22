import unittest
from datetime import datetime

import imap_processing.tests.ultra.data.mock_data as umd
import numpy as np
from astropy_healpix import HEALPix
from imap_processing.ena_maps.ena_maps import UltraPointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry

from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.ultra.l3.science.ultra_survival_probability import UltraSurvivalProbability


class TestUltraSurvivalProbability(unittest.TestCase):

    def test_ultra_survival_probability_pset_calls_super(self):
        l1cdataset = umd.mock_l1c_pset_product_healpix(nside=16)
        input_l1c_pset = _create_ultra_l1c_pset_from_xarray(l1cdataset)
        glows = _build_glows_l3e_ultra()
        l1cdataset.attrs = None

        prod = UltraSurvivalProbability(input_l1c_pset, glows)

        self.assertIsInstance(prod, UltraPointingSet)
        self.assertEqual(l1cdataset, prod.data)
        self.assertEqual(geometry.SpiceFrame.IMAP_DPS, prod.spice_reference_frame)


def _build_glows_l3e_ultra(survival_probabilities: np.ndarray = None, energies: np.ndarray = None, nside: int = 16):
    energies = energies or np.arange(20)
    num_pixels = 12 * (nside ** 2)
    survival_probabilities = survival_probabilities or np.random.rand(1, len(energies), num_pixels)
    healpix_pixels = np.arange(num_pixels)

    healpix_grid = HEALPix(nside)
    lon, lat = healpix_grid.healpix_to_lonlat(healpix_pixels)

    return UltraGlowsL3eData(
        epoch=datetime.now(),
        energy=energies,
        healpix_index=healpix_pixels,
        survival_probability=survival_probabilities,
        latitude=lat.value,
        longitude=lon.value
    )


def _create_ultra_l1c_pset_from_xarray(l1cdataset):
    l1c_energies = l1cdataset.coords[CoordNames.ENERGY.value].values
    l1c_exposure = np.repeat(l1cdataset["exposure_time"].values[np.newaxis, :], len(l1c_energies), 1)
    l1c_exposure = np.reshape(l1c_exposure, (1, len(l1c_energies), -1))

    l1cdataset["exposure_time"] = (
        [
            CoordNames.TIME.value,
            CoordNames.ENERGY.value,
            CoordNames.HEALPIX_INDEX.value
        ],
        l1c_exposure
    )

    input_l1c_pset = UltraL1CPSet(
        epoch=l1cdataset.coords[CoordNames.TIME.value].values[0],
        energy=l1cdataset.coords[CoordNames.ENERGY.value].values,
        counts=l1cdataset["counts"].values,
        exposure=l1cdataset["exposure_time"].values,
        healpix_index=l1cdataset.coords[CoordNames.HEALPIX_INDEX.value].values,
        latitude=l1cdataset[CoordNames.ELEVATION_L1C.value].values,
        longitude=l1cdataset[CoordNames.AZIMUTH_L1C.value].values,
        sensitivity=l1cdataset["sensitivity"].values
    )
    return input_l1c_pset


if __name__ == '__main__':
    unittest.main()
