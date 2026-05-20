import dataclasses
import logging
import shutil
from datetime import datetime
from typing import Self

from imap_data_access import ProcessingInputCollection, ScienceFilePath
from imap_data_access.processing_input import generate_imap_input, ScienceInput
from imap_processing.hit.l1b.constants import FILLVAL_INT64

import numpy as np
from imap_processing.cdf.utils import write_cdf as write_l2_cdf
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.lo.l3.lo_sp_initializer import LO_SP_MAP_KERNELS
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import furnish_spice_metakernel
from tests.test_helpers import get_test_data_path, get_run_local_data_path
from pathlib import Path
import pandas
import xarray as xr
from imap_l3_processing.constants import TT2000_EPOCH
import re
import imap_data_access

LO_ENERGIES_IN_KEV = np.array([16.33, 30.47, 55.76, 106.3, 200.0, 405.0, 787.3]) / 1000.0
LO_ENERGY_BIN_LOWERS = np.array([10.9, 20.4, 36.8, 71.6, 135.0, 269.0, 504.7]) / 1000.0
LO_ENERGY_BIN_UPPERS = np.array([21.7, 40.5, 74.7, 140.9, 265.0, 541.0, 1069.9]) / 1000.0

@dataclasses.dataclass
class LoProcessingInput:
    l2_cdf_path: Path
    repoints: list[int]
    start_date: datetime
    end_date: datetime

    @classmethod
    def read_from_txt_data(cls, input_path: Path, output_descriptor: str) -> Self:
        loaded_data = {}
        for data_file_path in input_path.iterdir():
            if fn_match := re.match("map_([a-zA-Z]+)_esa(\d{1}).csv", data_file_path.name):
                data_type, energy_level = fn_match.groups()
                if data_type not in loaded_data:
                    loaded_data[data_type] = np.full((1, 7, 60, 30), np.nan)

                esa_step = int(energy_level) - 1
                loaded_data[data_type][0, esa_step] = np.loadtxt(data_file_path, delimiter=",", skiprows=1).T

        manifest_df = pandas.read_csv(input_path / "map_l1b_manifest_esa1.csv")
        l1c_times = pandas.to_datetime(manifest_df["date_yyyymmdd"], format="%Y%m%d")

        start_date = min(l1c_times)
        end_date = max(l1c_times)

        start_date_nanoseconds = (start_date - TT2000_EPOCH).total_seconds() * 1e9
        end_date_nanoseconds = (end_date - TT2000_EPOCH).total_seconds() * 1e9

        longitude = np.linspace(3, 357, 60)
        latitude = np.linspace(-87, 87, 30)

        map_data_coords = ["epoch", "energy", "longitude", "latitude"]
        l2_dataset = xr.Dataset(
            data_vars={
                "ena_intensity": (map_data_coords, loaded_data["flux"]),
                "ena_intensity_stat_uncert": (map_data_coords, np.sqrt(loaded_data["fvar"])),
                "ena_intensity_sys_err": (map_data_coords, loaded_data["fser"]),
                "bg_intensity": (map_data_coords, loaded_data["bflux"]),
                "bg_intensity_stat_uncert": (map_data_coords, np.sqrt(loaded_data["bfvar"])),
                "bg_intensity_sys_err": (map_data_coords, np.full((1, 7, 60, 30), np.nan)),
                "exposure_factor": (map_data_coords, loaded_data["expo"]),
                "obs_date": (map_data_coords, np.full((1, 7, 60, 30), FILLVAL_INT64)),
                "obs_date_range": (map_data_coords, np.full((1, 7, 60, 30), np.nan)),
                "solid_angle": (["epoch", "longitude", "latitude"], np.full((1, 60, 30), np.nan)),
                "energy_delta_minus": (["energy"], LO_ENERGIES_IN_KEV - LO_ENERGY_BIN_LOWERS),
                "energy_delta_plus": (["energy"], LO_ENERGY_BIN_UPPERS - LO_ENERGIES_IN_KEV),
            },
            coords={
                "epoch": np.array([start_date_nanoseconds]),
                "energy": LO_ENERGIES_IN_KEV,
                "longitude": longitude,
                "latitude": latitude,
            },
        )

        variables_to_mask = [
            "ena_intensity",
            "ena_intensity_stat_uncert",
            "ena_intensity_sys_err",
            "bg_intensity",
            "bg_intensity_stat_uncert",
            "bg_intensity_sys_err",
        ]

        mask = l2_dataset["exposure_factor"] != 0.0
        for var in variables_to_mask:
            l2_dataset[var] = l2_dataset[var].where(mask, np.nan)

        skymap = RectangularSkyMap(6, SpiceFrame.ECLIPJ2000)
        skymap.max_epoch = int(end_date_nanoseconds)
        skymap.min_epoch = int(start_date_nanoseconds)

        l2_dataset_with_metadata = skymap.build_cdf_dataset(
            instrument="lo",
            level="l2",
            descriptor=output_descriptor,
            external_map_dataset=l2_dataset,
        )
        version = "001"
        l2_dataset_with_metadata.attrs["Data_version"] = version

        repoints = list(manifest_df["repoint"])

        l2_cdf_path = ScienceFilePath.generate_from_inputs(
            instrument="lo",
            data_level="l2",
            descriptor=output_descriptor,
            start_time=start_date.strftime("%Y%m%d"),
            version=f"v{version}",
            repointing=None,
        ).construct_path()

        if not l2_cdf_path.exists():
            l2_cdf_path = write_l2_cdf(l2_dataset_with_metadata)

        return cls(
            l2_cdf_path=l2_cdf_path,
            repoints=repoints,
            start_date=start_date,
            end_date=end_date,
        )

    def to_processing_input_collection(self, glows_input: list[Path]) -> ProcessingInputCollection:
        l1c_results = imap_data_access.query(instrument="lo", data_level="l1c", version="latest")
        l1c_file_names = [Path(l1c["file_path"]).name for l1c in l1c_results if l1c["repointing"] in self.repoints]

        glows_input = [Path(fp).name for fp in glows_input]

        inputs = [self.l2_cdf_path.name] + l1c_file_names + glows_input
        return ProcessingInputCollection(*(generate_imap_input(fn) for fn in inputs))

def filter_glows_data_for_lo_inputs(glows_input_dir: Path, repoints: list[int]) -> list[Path]:
    filtered_glows_data = []
    for fp in glows_input_dir.rglob("*.cdf"):
        science_file = ScienceFilePath(fp)
        if science_file.descriptor == "survival-probability-lo" and science_file.repointing in repoints:
            filtered_glows_data.append(fp)
    return filtered_glows_data

if __name__ == "__main__":
    logging.basicConfig(force=True, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    output_data_path = get_run_local_data_path("lo_txt_pipeline")
    # shutil.rmtree(output_data_path, ignore_errors=True)
    imap_data_access.config["DATA_DIR"] = output_data_path

    glows_data_dir = get_run_local_data_path("glows_l3bcde_with_prod_l2/imap/glows/l3e")

    cg_corrected_input_path = get_test_data_path("lo/lo_txt_pipeline/3S5_l1b_ram_maps")
    for pivot in (90, 105, 75):
        l2_descriptor = f"l{pivot:03d}-ena-h-sf-nsp-ram-hae-6deg-1yr"

        csv_maps_path = cg_corrected_input_path / "outdir" / f"pivot_{pivot}" / "maps"
        lo_txt_input = LoProcessingInput.read_from_txt_data(csv_maps_path, l2_descriptor)
        glows_input_paths = filter_glows_data_for_lo_inputs(glows_data_dir, lo_txt_input.repoints)

        for glows_input_path in glows_input_paths:
            glows_path_in_data_dir = ScienceFilePath(Path(glows_input_path).name).construct_path()
            glows_path_in_data_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(glows_input_path, glows_path_in_data_dir)

        sp_descriptor = l2_descriptor.replace("-nsp-", "-sp-")
        sp_processor = LoProcessor(
            dependencies=lo_txt_input.to_processing_input_collection(glows_input_paths),
            input_metadata=InputMetadata(
                instrument="lo",
                data_level="l3",
                start_date=lo_txt_input.start_date,
                end_date=lo_txt_input.end_date,
                version="v001",
                descriptor=sp_descriptor
            )
        )

        furnish_spice_metakernel(lo_txt_input.start_date, lo_txt_input.end_date, LO_SP_MAP_KERNELS)

        [sp_map] = sp_processor.process()
        print("Produced: ", sp_map)

        spx_descriptor = sp_descriptor.replace("-ena-", "-spx-")
        spx_processor = LoProcessor(
            dependencies=ProcessingInputCollection(ScienceInput(sp_map.name)),
            input_metadata=InputMetadata(
                instrument="lo",
                data_level="l3",
                start_date=lo_txt_input.start_date,
                end_date=lo_txt_input.end_date,
                version="v001",
                descriptor=spx_descriptor
            )
        )

        [spx_map] = spx_processor.process()
        print("Produced: ", spx_map)

    ram_nbs_input_path = get_test_data_path("lo/lo_txt_pipeline/3S5_l1b_ram_maps")
    for pivot in (90, 105, 75):
        l2_nbs_descriptor = f"l{pivot:03d}-enanbs-h-sf-nsp-ram-hae-6deg-1yr"

        csv_maps_path = ram_nbs_input_path / "outdir" / f"pivot_{pivot}" / "maps"
        lo_txt_input = LoProcessingInput.read_from_txt_data(csv_maps_path, l2_nbs_descriptor)

        spx_nbs_descriptor = l2_nbs_descriptor.replace("-enanbs-", "-spxnbs-")
        spx_nbs_processor = LoProcessor(
            dependencies=ProcessingInputCollection(ScienceInput(lo_txt_input.l2_cdf_path.name)),
            input_metadata=InputMetadata(
                instrument="lo",
                data_level="l3",
                start_date=lo_txt_input.start_date,
                end_date=lo_txt_input.end_date,
                version="v001",
                descriptor=spx_nbs_descriptor
            )
        )

        [spx_nbs_map] = spx_nbs_processor.process()
        print("Produced: ", spx_nbs_map)
