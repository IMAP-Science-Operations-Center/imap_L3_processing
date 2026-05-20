import dataclasses
import logging
import shutil
from datetime import datetime
from typing import Self, Optional

import imap_data_access
from imap_data_access import ProcessingInputCollection, ScienceFilePath
from imap_data_access.processing_input import ScienceInput, ProcessingInput
from imap_processing.hit.l1b.constants import FILLVAL_INT64

import numpy as np
from imap_processing.cdf.utils import write_cdf as write_l2_cdf
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.lo.l3.lo_sp_initializer import LO_SP_MAP_KERNELS
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import furnish_spice_metakernel
from tests.test_helpers import get_run_local_data_path
from pathlib import Path
import pandas
import xarray as xr
from imap_l3_processing.constants import TT2000_EPOCH
import re

LO_ENERGIES_IN_KEV = np.array([16.33, 30.47, 55.76, 106.3, 200.0, 405.0, 787.3]) / 1000.0
LO_ENERGY_BIN_LOWERS = np.array([10.9, 20.4, 36.8, 71.6, 135.0, 269.0, 504.7]) / 1000.0
LO_ENERGY_BIN_UPPERS = np.array([21.7, 40.5, 74.7, 140.9, 265.0, 541.0, 1069.9]) / 1000.0

@dataclasses.dataclass
class CsvNameToProduct:
    flux: str
    flux_variance: str
    flux_sys_err: str
    bg_flux: str
    bg_flux_variance: str
    bg_flux_sys_err: str
    exposure_factor: Optional[str]

    @classmethod
    def create_nbs_mapping(cls):
        return cls(
            flux="map_flux",
            flux_variance="map_fvar",
            flux_sys_err="map_fser",
            bg_flux="map_bflux",
            bg_flux_variance="map_bfvar",
            bg_flux_sys_err="map_bfunc",
            exposure_factor="map_expo",
        )

    @classmethod
    def create_cg_mapping(cls):
        return cls(
            flux="map_cgflux",
            flux_variance="map_cgfvar",
            flux_sys_err="map_cgfunc",
            bg_flux="bkg_cgflux",
            bg_flux_variance="bkg_cgfvar",
            bg_flux_sys_err="bkg_cgfunc",
            exposure_factor=None,
        )


@dataclasses.dataclass
class Manifest:
    repoints: list[int]
    start_date: datetime
    end_date: datetime

    @classmethod
    def load(cls, manifest_path: Path) -> Self:
        manifest_df = pandas.read_csv(manifest_path)
        l1c_times = pandas.to_datetime(manifest_df["date_yyyymmdd"], format="%Y%m%d")
        start_date = min(l1c_times)
        end_date = max(l1c_times)

        repoints = list(manifest_df["repoint"])

        return cls(
            repoints=repoints,
            start_date=start_date,
            end_date=end_date,
        )

@dataclasses.dataclass
class LoProcessingInput:
    repoints: list[int]
    start_date: datetime
    end_date: datetime
    version: int
    l2_descriptor: str
    dataset: xr.Dataset

    @staticmethod
    def load_data_dir(data_dir, **kwargs):
        temp_data = {}
        for data_file_path in data_dir.iterdir():
            if fn_match := re.match("([a-zA-Z]+_[a-zA-Z]+)_esa(\d{1}).csv", data_file_path.name):
                data_type, energy_level = fn_match.groups()
                if data_type not in temp_data:
                    temp_data[data_type] = np.full((1, 7, 60, 30), np.nan)

                esa_step = int(energy_level) - 1
                temp_data[data_type][0, esa_step] = np.loadtxt(data_file_path, delimiter=",", **kwargs).T
        return temp_data

    @classmethod
    def load(cls, manifest: Manifest, l2_descriptor: str, version: int, input_path: Path, name_mapping: CsvNameToProduct, **additional_map_data) -> Self:
        loaded_data = {
            **LoProcessingInput.load_data_dir(input_path / "maps", skiprows=1),
            **LoProcessingInput.load_data_dir(input_path / "masked_maps")
        }

        start_date_nanoseconds = (manifest.start_date - TT2000_EPOCH).total_seconds() * 1e9
        end_date_nanoseconds = (manifest.end_date - TT2000_EPOCH).total_seconds() * 1e9

        skymap = RectangularSkyMap(6, SpiceFrame.ECLIPJ2000)
        skymap.max_epoch = int(end_date_nanoseconds)
        skymap.min_epoch = int(start_date_nanoseconds)

        map_data_coords = ["epoch", "energy", "longitude", "latitude"]
        l2_dataset = xr.Dataset(
            data_vars={
                "ena_intensity": (map_data_coords, loaded_data[name_mapping.flux]),
                "ena_intensity_stat_uncert": (map_data_coords, np.sqrt(loaded_data[name_mapping.flux_variance])),
                "ena_intensity_sys_err": (map_data_coords, loaded_data[name_mapping.flux_sys_err]),
                "bg_intensity": (map_data_coords, loaded_data[name_mapping.bg_flux]),
                "bg_intensity_stat_uncert": (map_data_coords, np.sqrt(loaded_data[name_mapping.bg_flux_variance])),
                "bg_intensity_sys_err": (map_data_coords, loaded_data[name_mapping.bg_flux_sys_err]),

                "obs_date": (map_data_coords, np.full((1, 7, 60, 30), FILLVAL_INT64)),
                "obs_date_range": (map_data_coords, np.full((1, 7, 60, 30), np.nan)),
                "solid_angle": (["epoch", "longitude", "latitude"], np.full((1, 60, 30), np.nan)),
                "energy_delta_minus": (["energy"], LO_ENERGIES_IN_KEV - LO_ENERGY_BIN_LOWERS),
                "energy_delta_plus": (["energy"], LO_ENERGY_BIN_UPPERS - LO_ENERGIES_IN_KEV),
            },
            coords={
                "epoch": np.array([start_date_nanoseconds]),
                "energy": LO_ENERGIES_IN_KEV,
                "longitude": skymap.sky_grid.az_bin_midpoints,
                "latitude": skymap.sky_grid.el_bin_midpoints,
            },
        )
        if name_mapping.exposure_factor is not None:
            l2_dataset = l2_dataset.assign(
                exposure_factor=(map_data_coords, loaded_data[name_mapping.exposure_factor])
            )

        l2_dataset = l2_dataset.assign(**additional_map_data)

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

        l2_dataset_with_metadata = skymap.build_cdf_dataset(
            instrument="lo",
            level="l2",
            descriptor=l2_descriptor,
            external_map_dataset=l2_dataset,
        )
        l2_dataset_with_metadata.attrs["Data_version"] = f"{version:03}"

        return cls(
            dataset=l2_dataset_with_metadata,
            l2_descriptor=l2_descriptor,
            version=version,
            **dataclasses.asdict(manifest)
        )

    def get_spx_dependencies(self):
        l2_path = write_l2_cdf(self.dataset)
        return ProcessingInputCollection(ScienceInput(l2_path.name))

    def get_glows_paths(self, glows_data_dir: Path):
        glows_paths = []
        for fp in glows_data_dir.rglob("*.cdf"):
            science_file = ScienceFilePath(fp)
            if science_file.descriptor == "survival-probability-lo" and science_file.repointing in self.repoints:
                glows_paths.append(fp)
        return glows_paths

    def get_survival_corrected_dependencies(self, glows_data_dir: Path):
        l2_path = write_l2_cdf(self.dataset)

        l1c_results = imap_data_access.query(instrument="lo", data_level="l1c", version="latest")
        l1c_paths = [Path(l1c["file_path"]) for l1c in l1c_results if l1c["repointing"] in self.repoints]
        glows_paths = self.get_glows_paths(glows_data_dir)

        return ProcessingInputCollection(*(ScienceInput(p.name) for p in [l2_path, *glows_paths, *l1c_paths]))

    def make_l3_input_metadata(self, l3_descriptor: str) -> InputMetadata:
        return InputMetadata(
            instrument="lo",
            data_level="l3",
            start_date=self.start_date,
            end_date=self.end_date,
            version="v001",
            descriptor=l3_descriptor
        )

if __name__ == "__main__":
    logging.basicConfig(force=True, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    output_data_path = get_run_local_data_path("lo_txt_pipeline")
    shutil.rmtree(output_data_path / "imap" / "lo" / "l2", ignore_errors=True)
    shutil.rmtree(output_data_path / "imap" / "lo" / "l3", ignore_errors=True)
    imap_data_access.config["DATA_DIR"] = output_data_path

    lo_input_data_dir = get_run_local_data_path("input_lo_txt_pipeline")
    glows_data_dir = lo_input_data_dir / "glows"
    cg_corrected_input_path = lo_input_data_dir / "3S8_l1b_cg_corrected"
    ram_nbs_input_path = lo_input_data_dir / "3S5_l1b_ram_maps"

    combined_inputs = []
    start_dates = []
    end_dates = []
    for pivot in (90, 105, 75):
        manifest_path = ram_nbs_input_path / "outdir" / f"pivot_{pivot}" / "maps" / "map_l1b_manifest_esa1.csv"
        manifest = Manifest.load(manifest_path)

        nbs_processing_input = LoProcessingInput.load(
            manifest,
            f"l{pivot:03d}-enanbs-h-sf-nsp-ram-hae-6deg-1yr",
            1,
            ram_nbs_input_path / "outdir" / f"pivot_{pivot}",
            CsvNameToProduct.create_nbs_mapping()
        )

        [spx_nbs_map] = LoProcessor(
            input_metadata=nbs_processing_input.make_l3_input_metadata(f"l{pivot:03d}-spxnbs-h-sf-nsp-ram-hae-6deg-1yr"),
            dependencies=nbs_processing_input.get_spx_dependencies(),
        ).process()
        print("Produced: ", spx_nbs_map)

        cg_processing_input = LoProcessingInput.load(
            manifest,
            f"l{pivot:03d}-ena-h-hf-nsp-ram-hae-6deg-1yr",
            1,
            cg_corrected_input_path / "outdir" / f"pivot_{pivot}",
            CsvNameToProduct.create_cg_mapping(),
            exposure_factor=nbs_processing_input.dataset["exposure_factor"]
        )

        for glows_input_path in cg_processing_input.get_glows_paths(glows_data_dir):
            glows_path_in_data_dir = ScienceFilePath(Path(glows_input_path).name).construct_path()
            glows_path_in_data_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(glows_input_path, glows_path_in_data_dir)

        furnish_spice_metakernel(cg_processing_input.start_date, cg_processing_input.end_date, LO_SP_MAP_KERNELS)

        [sp_map] = LoProcessor(
            input_metadata=cg_processing_input.make_l3_input_metadata(f"l{pivot:03d}-ena-h-hf-sp-ram-hae-6deg-1yr"),
            dependencies=cg_processing_input.get_survival_corrected_dependencies(glows_data_dir),
        ).process()
        print("Produced: ", sp_map)

        combined_inputs.append(sp_map)
        start_dates.append(cg_processing_input.start_date)
        end_dates.append(cg_processing_input.end_date)

        [spx_map] = LoProcessor(
            input_metadata=cg_processing_input.make_l3_input_metadata(f"l{pivot:03d}-spx-h-hf-sp-ram-hae-6deg-1yr"),
            dependencies=ProcessingInputCollection(ScienceInput(sp_map.name)),
        ).process()
        print("Produced: ", spx_map)

    combined_descriptor = "ilo-ena-h-hf-sp-ram-hae-6deg-1yr"
    combined_start_date = min(start_dates)
    combined_end_date = max(end_dates)
    combined_processor = LoProcessor(
        dependencies=ProcessingInputCollection(*[ScienceInput(map_path.name) for map_path in combined_inputs]),
        input_metadata=InputMetadata(
            instrument="lo",
            data_level="l3",
            start_date=combined_start_date,
            end_date=combined_end_date,
            version="v001",
            descriptor=combined_descriptor
        )
    )

    [combined_sp_map] = combined_processor.process()
    print("Produced: ", combined_sp_map)

    combined_spx_descriptor = combined_descriptor.replace("-ena-", "-spx-")
    combined_spx_processor = LoProcessor(
        dependencies=ProcessingInputCollection(ScienceInput(combined_sp_map.name)),
        input_metadata=InputMetadata(
            instrument="lo",
            data_level="l3",
            start_date=combined_start_date,
            end_date=combined_end_date,
            version="v001",
            descriptor=combined_spx_descriptor
        )
    )

    [combined_spx_map] = combined_spx_processor.process()
    print("Produced: ", combined_spx_map)



