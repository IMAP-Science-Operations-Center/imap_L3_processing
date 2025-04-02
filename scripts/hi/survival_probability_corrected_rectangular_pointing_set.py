import unittest
from datetime import datetime
from pathlib import Path
from time import strftime

import imap_data_access
import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularPointingSet, RectangularSkyMap, AbstractSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.spatial_utils import AzElSkyGrid
from imap_processing.spice import geometry

from spacepy.pycdf import CDF
from spiceypy import spiceypy
import enum

from scripts.glows.create_example_glows_l3e_survival_probabilities_cdf import survival_probabilities


class IncludedSensors(enum.Enum):
    Hi90 = "90"
    Hi45 = "45"
    Combined = "combined"


class HiSurvivalProbabilityPointingSet(RectangularPointingSet):
    def __init__(self, l1c_dataset: xr.Dataset, glows_data_cdf: np.array):
        skygrid = AzElSkyGrid(spacing_deg=1)

        # super().__init__(l1c_dataset=l1c_dataset)
        # filepaths = imap_data_access.query(instrument="GLOWS",data_level="l3e",descriptor="survival-probabilities-hi",
        #                                    start_date=self.epoch.strftime("%Y%m%d"), end_date=self.epoch.strftime("%Y%m%d"))
        # glows_data_path = imap_data_access.download(filepaths[0]['file_path'])
        # glows_data_cdf = CDF(str(glows_data_path))
        # assumes l1c_dataset['counts'].shape == (epoch, lon, lat, energy) -> (1, 3600, 2, 9)

        hi90_elevation_bin = np.digitize(0, skygrid.el_bin_edges)
        hi45_elevation_bin = np.digitize(-45, skygrid.el_bin_edges)

        survival_probabilities = np.zeros((1, 9, len(skygrid.az_bin_midpoints), len(skygrid.el_bin_midpoints)))
        exposure = np.zeros((1, 9, len(skygrid.az_bin_midpoints), len(skygrid.el_bin_midpoints)))
        for spin_angle_index in range(len(glows_data_cdf['spin_angle'][...])):
            survival_probabilities[0, :, spin_angle_index, hi45_elevation_bin] = 10 ** np.interp(
                np.log10(l1c_dataset['energy']),
                glows_data_cdf['probability_of_survival'][
                    spin_angle_index],
                np.log10(glows_data_cdf['energy']))
            survival_probabilities[0, :, spin_angle_index, hi90_elevation_bin] = 10 ** np.interp(
                np.log10(l1c_dataset['energy']),
                glows_data_cdf['probability_of_survival'][
                    spin_angle_index],
                np.log10(glows_data_cdf['energy']))

        exposure[0, :, :, hi45_elevation_bin] = l1c_dataset['exposure'].values
        exposure[0, :, :, hi90_elevation_bin] = l1c_dataset['exposure'].values

        exposure_weighted_survival_probabilities = exposure * survival_probabilities

        survival_prob_dataset = xr.Dataset(
            {
                "exposure_weighted_survival_probabilities": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY.value,
                        CoordNames.AZIMUTH_L1C.value,
                        CoordNames.ELEVATION_L1C.value,
                    ],
                    exposure_weighted_survival_probabilities
                ),
                "exposure": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY.value,
                        CoordNames.AZIMUTH_L1C.value,
                        CoordNames.ELEVATION_L1C.value,
                    ],
                    exposure
                ),
            },
            coords={
                CoordNames.TIME.value: glows_data_cdf['epoch'][...],
                CoordNames.ENERGY.value: l1c_dataset["energy"].values,
                CoordNames.AZIMUTH_L1C.value: skygrid.az_bin_midpoints,
                CoordNames.ELEVATION_L1C.value: skygrid.el_bin_midpoints,
            },
            attrs={
                "Logical_file_id": (
                    f"imap_ultra_l1c_90sensor-pset_{datetime.now().strftime("%Y%m%d")}"
                )
            },
        )
        super().__init__(survival_prob_dataset)


class HiSurvivalProbabilitySkyMap(RectangularSkyMap):
    def __init__(self, glows_pointing_set: list[HiSurvivalProbabilityPointingSet]):
        super().__init__(2, geometry.SpiceFrame.IMAP_DPS)

        for pset in glows_pointing_set:
            self.project_pset_values_to_map(pset, ["exposure_weighted_survival_probabilities", "exposure"])

        self.data_1d["exposure_weighted_survival_probabilities"] /= self.data_1d["exposure"]
        print("breakpoint")

        # self.data_1d["intensity_survival_corrected"].values = l2_l3_map.data_1d["intensity"].values / self.data_1d[
        #     "survival_prob_exposure_weighted"]


if __name__ == '__main__':
    includedSensors = IncludedSensors.Hi90

    kernels = Path(__file__).parent.parent.parent.joinpath("spice_kernels")
    for file in kernels.iterdir():
        spiceypy.furnsh(str(file))

    # find actual exposure shape
    exposure = np.full(shape=(1, 9, 360), fill_value=1)
    l1c_dataset = xr.Dataset({
        "exposure": xr.DataArray(data=exposure, dims=["epoch", "energy", "spin_angle"]),
        "counts": xr.DataArray(data=np.full_like(exposure, 1), dims=["epoch", "energy", "spin_angle"]),
    },
        coords={
            "spin_angle": np.arange(360) + 0.05,
            "energy": np.geomspace(0.37498966, 19.582634 - 5, 9),
            "epoch": [datetime(2009, 1, 1)]
        })

    glows_cdf = CDF(
        r"C:\Users\Harrison\Development\imap_L3_processing\tests\test_data\glows\imap_glows_l3e_survival-probabilities-hi_20250324_v001.cdf")

    elongations = glows_cdf['elongation'][...]
    survival_probabilities_combined = glows_cdf["probability_of_survival"][...]
    survival_probabilities_135 = survival_probabilities[0]
    survival_probabilities_90 = survival_probabilities[1]

    glows_pointing_set = [HiSurvivalProbabilityPointingSet(l1c_dataset, glows_cdf)]
    survival_probability_skymap = HiSurvivalProbabilitySkyMap(glows_pointing_set)

    output = survival_probability_skymap.to_dataset()
