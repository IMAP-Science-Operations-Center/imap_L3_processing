import numpy as np
import spiceypy
from astropy.units import Quantity
from astropy_healpix import HEALPix, npix_to_nside
from imap_processing.ena_maps.ena_maps import UltraPointingSet, HealpixSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from xarray import Dataset

from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData


class UltraSurvivalProbability(UltraPointingSet):
    def __init__(self, l1c_pset: UltraL1CPSet, l3e_glows: UltraGlowsL3eData):
        super().__init__(l1c_pset.to_xarray(), geometry.SpiceFrame.IMAP_DPS)

        l1c_epoch_in_et = spiceypy.unitim(self.data.coords[CoordNames.TIME.value].values[0] / ONE_SECOND_IN_NANOSECONDS,
                                          "TT", "ET")
        rotated_az_el_points = geometry.frame_transform_az_el(l1c_epoch_in_et, self.az_el_points,
                                                              geometry.SpiceFrame.IMAP_DPS,
                                                              geometry.SpiceFrame.ECLIPJ2000)
        glows_nside = npix_to_nside(len(l3e_glows.healpix_index))
        glows_healpix = HEALPix(nside=glows_nside)

        npixels = len(l1c_pset.healpix_index)

        spatially_interpolated_sp = np.zeros((len(l3e_glows.energy), npixels))

        for energy_index in range(len(l3e_glows.energy)):
            spatially_interpolated_sp[energy_index, :] = glows_healpix.interpolate_bilinear_lonlat(
                Quantity(rotated_az_el_points[:, 0], unit='deg'),
                Quantity(rotated_az_el_points[:, 1], unit='deg'),
                l3e_glows.survival_probability[0, energy_index, :])

        energy_interpolated_sp = np.zeros((len(l1c_pset.energy), npixels))
        for healpix_index in range(npixels):
            energy_interpolated_sp[:, healpix_index] = np.interp(
                np.log10(l1c_pset.energy),
                np.log10(l3e_glows.energy),
                spatially_interpolated_sp[:, healpix_index]
            )

        self.data["survival_probability_times_exposure"] = (
            [
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA.value,
                CoordNames.HEALPIX_INDEX.value
            ],
            np.array([energy_interpolated_sp] * l1c_pset.exposure)
        )


class UltraSurvivalProbabilitySkyMap(HealpixSkyMap):
    def __init__(self, sp: list[UltraSurvivalProbability], spice_frame: geometry.SpiceFrame, nside: int):
        super().__init__(nside, spice_frame)
        for sp_pset in sp:
            self.project_pset_values_to_map(sp_pset, ["survival_probability_times_exposure", "exposure_time"])

        self.data_1d = Dataset({
            "exposure_weighted_survival_probabilities": self.data_1d["survival_probability_times_exposure"] /
                                                        self.data_1d["exposure_time"]
        })
