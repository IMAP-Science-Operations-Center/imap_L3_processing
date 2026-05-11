from imap_l3_processing.hi.utils import read_glows_l3e_data
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, IntensityMapData, RectangularCoords, InputRectangularPointingSet, GlowsL3eRectangularMapInputData, RectangularIntensityDataProduct
import numpy as np
from imap_l3_processing.maps.survival_probability_processing import process_survival_probabilities
from imap_l3_processing.utils import save_data
from imap_processing.ena_maps.ena_maps import RectangularPointingSet
from imap_processing.spice.geometry import SpiceFrame
from tests.test_helpers import get_test_data_path, get_run_local_data_path
from pathlib import Path
import pandas

class LoTxtToCDFConverter:
    def create_map_data(self, input_path: Path) -> RectangularIntensityMapData:
        data_types = [
            "flux",
            "fvar",
            "bflux",
            "bfvar",
            "expo",
        ]

        loaded_data = {}

        for data_type in data_types:
            loaded_data[data_type] = np.full((1, 7, 60, 30), np.nan)

            for data_file_path in input_path.glob(f"*_{data_type}.txt"):
                esa_step = int(data_file_path.name.split('-')[1][0]) - 1
                loaded_data[data_type][0, esa_step] = np.loadtxt(data_file_path, skiprows=28).T

        longitude = np.linspace(3, 357, 60)
        longitude_delta = np.full((60,), 3)
        latitude = np.linspace(-87, 87, 30)
        latitude_delta = np.full((30,), 3)

        intensity_data = IntensityMapData(
            ena_intensity = loaded_data["flux"],
            ena_intensity_stat_uncert = loaded_data["fvar"],
            ena_intensity_sys_err = loaded_data["fvar"],
            bg_intensity = loaded_data["bflux"],
            bg_intensity_stat_uncert = loaded_data["bfvar"],
            bg_intensity_sys_err = loaded_data["bfvar"],
            epoch=np.array([0]),
            epoch_delta=np.array([0]),
            energy=np.arange(1, 8, 1),
            energy_delta_plus=np.full((7,), 1.),
            energy_delta_minus=np.full((7,), 0),
            energy_label=[f"ESA {i}" for i in range(1, 8)],
            latitude=latitude,
            longitude=longitude,
            exposure_factor=loaded_data["expo"],
            obs_date=np.full((1, 7, 60, 30), np.nan),
            obs_date_range=np.full((1, 7, 60, 30), np.nan),
            solid_angle=np.full((1, 7, 60, 30), np.nan),
        )
        coords = RectangularCoords(
            latitude_delta=latitude_delta,
            latitude_label=[str(i) for i in latitude],
            longitude_delta=longitude_delta,
            longitude_label=[str(i) for i in longitude],
        )

        return RectangularIntensityMapData(
            intensity_map_data=intensity_data,
            coords=coords,
        )

    def create_pset_data(self, input_path: Path) -> InputRectangularPointingSet:
        reshaped_pset = pandas.read_csv(input_path).pivot(index="esa_level", columns="bins")

        assert np.all()

        hae_longitude = reshaped_pset["bin_ecl_lon"].to_numpy()
        hae_latitude = reshaped_pset["bin_ecl_lat"].to_numpy()
        return InputRectangularPointingSet(
            epoch=0,
            epoch_delta=0,
            epoch_j2000=0,
            repointing=217,
            exposure_times=reshaped_pset["expo"].to_numpy()[np.newaxis, ...],
            esa_energy_step=np.arange(1, 8, 1),
            pointing_start_met=0,
            pointing_end_met=0,
            hae_longitude=hae_longitude[:1, :],
            hae_latitude=hae_latitude[:1, :]
        )

if __name__ == "__main__":
    map_descriptor = "l090-ena-h-sf-sp-ram-hae-6deg-1yr"
    spice_frame = SpiceFrame.ECLIPJ2000

    l2_path = get_test_data_path("lo/lo_txt_pipeline/output/soc")
    l1c_path = get_test_data_path("lo/lo_txt_pipeline/output/map.csv")
    glows_data_path = get_run_local_data_path("glows_l3bcde_with_prod_l2/imap/glows/l3e/2025/12/imap_glows_l3e_survival-probability-lo_20251212-repoint00076_v001.cdf")

    converter = LoTxtToCDFConverter()
    l2_data = converter.create_map_data(l2_path)
    l1c_data = converter.create_pset_data(l1c_path)
    glows_data = read_glows_l3e_data(glows_data_path)
    glows_data.repointing = 217

    lo_survival_deps = HiLoL3SurvivalDependencies(
        l2_data=l2_data,
        l1c_data=[l1c_data],
        glows_l3e_data=[glows_data],
        l2_map_descriptor_parts=parse_map_descriptor(map_descriptor)
    )

    survival_corrected_data = process_survival_probabilities(lo_survival_deps, spice_frame_name=spice_frame)

    input_metadata = InputMetadata(
        instrument="lo",
        data_level="l3",
        start_date="20260101",
        end_date="20270101",
        version="v001",
        descriptor=map_descriptor
    )

    data_product = RectangularIntensityDataProduct(
        input_metadata=input_metadata,
        data=survival_corrected_data,
        spice_frame_name=spice_frame
    )

    print(save_data(data_product))

