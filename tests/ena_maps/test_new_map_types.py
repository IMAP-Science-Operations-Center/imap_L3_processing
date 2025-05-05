from unittest import TestCase

import numpy as np
from cdflib.xarray import cdf_to_xarray
from imap_processing.cdf.utils import write_cdf
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame
from xarray import Dataset

from imap_l3_processing import spice_wrapper
from imap_l3_processing.ena_maps.new_map_types import DerivedPointingSet, LinearInterpolateInLogOperation, \
    HiSurvivalCorrection, RectangularProtomap
from imap_l3_processing.hi.l3.utils import Sensor
from tests.test_helpers import get_test_data_path


class TestRebinPointingSetByEnergy(TestCase):
    class SimplePSet(DerivedPointingSet):
        def __init__(self, data: Dataset):
            super().__init__(data)

    def test_transform(self):
        reference_energies = np.array([10, 1000])
        pset_energies = np.array([1, 100, 100000])

        op = LinearInterpolateInLogOperation(CoordNames.ENERGY, reference_energies)

        dataset_to_rebin = Dataset(
            {"survival_probability": (
                [CoordNames.TIME.value, CoordNames.ENERGY.value, CoordNames.AZIMUTH_L1C.value],
                [[[1, 2, 3], [3, 8, 5], [6, 14, 20]]])},
            coords={CoordNames.TIME.value: [1], CoordNames.ENERGY.value: pset_energies,
                    CoordNames.AZIMUTH_L1C.value: [0, 120, 240]})

        transformed_pset = op.transform(self.SimplePSet(dataset_to_rebin))

        np.testing.assert_array_equal(transformed_pset.data["survival_probability"].values, [[[2, 5, 4], [4, 10, 10]]])
        np.testing.assert_array_equal(transformed_pset.data[CoordNames.ENERGY.value].values, reference_energies)

    def test_pointing_set_operations_integration(self):
        spice_wrapper.ensure_furnished()

        l1c_data_path = get_test_data_path("hi/imap_hi_l1c_90sensor-pset_20250415_v001.cdf")
        l1c_data = cdf_to_xarray(str(l1c_data_path), to_datetime=False).rename_vars(
            {"exposure_times": "exposure_factor"})

        l1c_pset = DerivedPointingSet(l1c_data, SpiceFrame.IMAP_DPS)

        glows_file_path = str(
            get_test_data_path("hi/imap_glows_l3e_survival-probabilities-hi-90-with-energy_20250416_v001.cdf"))
        glows_pset = DerivedPointingSet(
            cdf_to_xarray(glows_file_path, to_datetime=False).rename({"spin_angle": CoordNames.AZIMUTH_L1C.value}),
            SpiceFrame.IMAP_DPS)

        hi_energies = np.array([0.5, 0.75, 1.13, 1.68, 2.52, 3.75, 5.62, 8.42, 12.65])

        one_spatial_dim_intensity = np.full((1, 9, 90 * 45), 1.0)

        map_dimensions = [CoordNames.TIME.value,
                          CoordNames.ENERGY.value,
                          CoordNames.GENERIC_PIXEL.value]

        l2_dataset_1d = Dataset({
            "ena_intensity": (map_dimensions, one_spatial_dim_intensity),
            "ena_intensity_stat_unc": (map_dimensions, one_spatial_dim_intensity + 22),
            "ena_intensity_sys_err": (map_dimensions, one_spatial_dim_intensity + 19),
        },
            coords={
                CoordNames.TIME.value: l1c_pset.data["epoch"].values,
                CoordNames.ENERGY.value: hi_energies,
                CoordNames.GENERIC_PIXEL.value: np.arange(0, 90 * 45)
            })

        l2_dataset = RectangularProtomap(spacing_deg=4, spice_frame=SpiceFrame.ECLIPJ2000)
        l2_dataset.data_1d = l2_dataset_1d

        map = HiSurvivalCorrection().survival_correct(Sensor.Hi90, [l1c_pset], l2_dataset, [glows_pset])

        resulting_dataset = map.to_dataset()
        resulting_dataset.attrs["Logical_source"] = ""

        print(write_cdf(resulting_dataset))
