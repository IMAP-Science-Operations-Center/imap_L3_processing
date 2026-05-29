import dataclasses
import logging
import shutil
from datetime import datetime
from typing import Self, Optional

import imap_data_access
from imap_data_access import ProcessingInputCollection, ScienceFilePath
from imap_data_access.processing_input import ScienceInput
from imap_processing.hit.l1b.constants import FILLVAL_INT64

import numpy as np
from imap_processing.cdf.utils import write_cdf as write_l2_cdf
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.spice.geometry import SpiceFrame
from spacepy.pycdf import CDF

from imap_l3_processing.lo.l3.lo_sp_initializer import LO_SP_MAP_KERNELS
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import furnished_metakernel
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
    l1b_filenames: list[str]
    start_date: datetime
    end_date: datetime

    @classmethod
    def load(cls, manifest_path: Path) -> Self:
        manifest_df = pandas.read_csv(manifest_path)
        l1c_times = pandas.to_datetime(manifest_df["date_yyyymmdd"], format="%Y%m%d")
        start_date = min(l1c_times)
        end_date = max(l1c_times)

        return cls(
            repoints=list(manifest_df["repoint"]),
            l1b_filenames=list(manifest_df["l1b_filename"]),
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
    l2_cdf_path: Path

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
    def load(cls,
         manifest: Manifest,
         l2_descriptor: str,
         version: int,
         input_path: Path,
         name_mapping: CsvNameToProduct,
         use_masked_data: bool,
         **additional_map_data
    ) -> Self:
        if use_masked_data:
            loaded_data = {
                **LoProcessingInput.load_data_dir(input_path / "maps", skiprows=1),
                **LoProcessingInput.load_data_dir(input_path / "masked_maps")
            }
        else:
            loaded_data = {
                **LoProcessingInput.load_data_dir(input_path / "maps", skiprows=1),
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
        l2_dataset_with_metadata.attrs["Parents"] = manifest.l1b_filenames

        return cls(
            dataset=l2_dataset_with_metadata,
            l2_descriptor=l2_descriptor,
            version=version,
            l2_cdf_path=write_l2_cdf(l2_dataset_with_metadata),
            repoints=manifest.repoints,
            start_date=manifest.start_date,
            end_date=manifest.end_date,
        )

    def get_spx_dependencies(self):
        return ProcessingInputCollection(ScienceInput(self.l2_cdf_path.name))

    def get_survival_corrected_dependencies(self, l1c_files: list[Path]):
        # l1c_results = imap_data_access.query(instrument="lo", data_level="l1c", version="latest")
        # l1c_inputs = [Path(l1c["file_path"]) for l1c in l1c_results if l1c["repointing"] in self.repoints]
        l1c_inputs = [p for p in l1c_files if ScienceFilePath(p.name).repointing in self.repoints]

        glows_query_results = imap_data_access.query(
            instrument="glows",
            data_level="l3e",
            descriptor="survival-probability-lo",
            version="latest",
        )
        glows_paths = [
            Path(glows["file_path"]) for glows in glows_query_results if int(glows["repointing"]) in self.repoints
        ]

        return ProcessingInputCollection(*(ScienceInput(p.name) for p in [self.l2_cdf_path, *glows_paths, *l1c_inputs]))

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
    cg_corrected_input_path = lo_input_data_dir / "3S8_l1b_cg_corrected"
    ram_nbs_input_path = lo_input_data_dir / "3S5_l1b_ram_maps"

    l1c_paths = []
    local_l1c_input = lo_input_data_dir / "l1c"
    for file in local_l1c_input.rglob("*.cdf"):
        sfp = ScienceFilePath(file.name)

        output_path = sfp.construct_path()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(file, output_path)
        l1c_paths.append(output_path)

    output_maps = []
    for masked, descriptor_suffix in [(False, ""), (True, "Masked")]:

        start_dates = []
        end_dates = []
        for pivot in (90, 105, 75):
            manifest_path = ram_nbs_input_path / "outdir" / f"pivot_{pivot}" / "maps" / "map_l1b_manifest_esa1.csv"
            manifest = Manifest.load(manifest_path)

            start_dates.append(manifest.start_date)
            end_dates.append(manifest.end_date)

            nbs_processing_input = LoProcessingInput.load(
                manifest,
                f"l{pivot:03d}-enanbs{descriptor_suffix}-h-sf-nsp-ram-hae-6deg-6mo",
                1,
                ram_nbs_input_path / "outdir" / f"pivot_{pivot}",
                CsvNameToProduct.create_nbs_mapping(),
                masked,
            )

            output_maps.append(nbs_processing_input.l2_cdf_path)

            [spx_nbs_map] = LoProcessor(
                input_metadata=nbs_processing_input.make_l3_input_metadata(f"l{pivot:03d}-spxnbs{descriptor_suffix}-h-sf-nsp-ram-hae-6deg-6mo"),
                dependencies=nbs_processing_input.get_spx_dependencies(),
            ).process()
            print("Produced: ", spx_nbs_map)

            output_maps.append(spx_nbs_map)

            cg_processing_input = LoProcessingInput.load(
                manifest,
                f"l{pivot:03d}-ena{descriptor_suffix}-h-hf-nsp-ram-hae-6deg-6mo",
                1,
                cg_corrected_input_path / "outdir" / f"pivot_{pivot}",
                CsvNameToProduct.create_cg_mapping(),
                masked,
                exposure_factor=nbs_processing_input.dataset["exposure_factor"]
            )

            output_maps.append(cg_processing_input.l2_cdf_path)

            [spx_cg_nsp_map] = LoProcessor(
                input_metadata=cg_processing_input.make_l3_input_metadata(f"l{pivot:03d}-spx{descriptor_suffix}-h-hf-nsp-ram-hae-6deg-6mo"),
                dependencies=cg_processing_input.get_spx_dependencies()
            ).process()

            output_maps.append(spx_cg_nsp_map)

            with furnished_metakernel(cg_processing_input.start_date, cg_processing_input.end_date, LO_SP_MAP_KERNELS):
                [sp_map] = LoProcessor(
                    input_metadata=cg_processing_input.make_l3_input_metadata(f"l{pivot:03d}-ena{descriptor_suffix}-h-hf-sp-ram-hae-6deg-6mo"),
                    dependencies=cg_processing_input.get_survival_corrected_dependencies(l1c_paths),
                ).process()
                print("Produced: ", sp_map)

                output_maps.append(sp_map)


            [spx_cg_sp_map] = LoProcessor(
                input_metadata=cg_processing_input.make_l3_input_metadata(f"l{pivot:03d}-spx{descriptor_suffix}-h-hf-sp-ram-hae-6deg-6mo"),
                dependencies=ProcessingInputCollection(ScienceInput(sp_map.name)),
            ).process()
            print("Produced: ", spx_cg_sp_map)

            output_maps.append(spx_cg_sp_map)

    maps_needed_for_combination = [
        ("ilo-enaMasked-h-hf-sp-ram-hae-6deg-6mo", [
            "l075-enaMasked-h-hf-sp-ram-hae-6deg-6mo",
            "l090-enaMasked-h-hf-sp-ram-hae-6deg-6mo",
            "l105-enaMasked-h-hf-sp-ram-hae-6deg-6mo"
        ]),
        ("ilo-enaMasked-h-hf-nsp-ram-hae-6deg-6mo", [
            "l075-enaMasked-h-hf-nsp-ram-hae-6deg-6mo",
            "l090-enaMasked-h-hf-nsp-ram-hae-6deg-6mo",
            "l105-enaMasked-h-hf-nsp-ram-hae-6deg-6mo"
        ]),
        ("ilo-enanbsMasked-h-sf-nsp-ram-hae-6deg-6mo", [
            "l075-enanbsMasked-h-sf-nsp-ram-hae-6deg-6mo",
            "l090-enanbsMasked-h-sf-nsp-ram-hae-6deg-6mo",
            "l105-enanbsMasked-h-sf-nsp-ram-hae-6deg-6mo"
        ])
    ]

    combined_start_date = min(start_dates)
    combined_end_date = max(end_dates)
    for combined_descriptor, combined_dependencies in maps_needed_for_combination:
        combined_inputs = []
        for output_map in output_maps:
            if any([f"_{combined_dep}_" in output_map.name for combined_dep in combined_dependencies]):
                combined_inputs.append(output_map)

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

        output_maps.append(combined_sp_map)

        combined_spx_descriptor = combined_descriptor.replace("-enaMasked-", "-spxMasked-")
        combined_spx_descriptor = combined_spx_descriptor.replace("-enanbsMasked-", "-spxnbsMasked-")

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

        output_maps.append(combined_spx_map)

    release_directory = get_run_local_data_path("IMAP-Lo May 29th 2026 Maps")
    for generated_path in output_maps:
        science_file_path = ScienceFilePath(generated_path.name)

        new_name = (
            "_".join(
                [
                    "imap",
                    science_file_path.instrument,
                    science_file_path.data_level,
                    science_file_path.descriptor + "-INITIAL",
                    science_file_path.start_date,
                    science_file_path.version,
                    ]
            )
            + ".cdf"
        )

        output_dir = release_directory / science_file_path.data_level
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / new_name

        with CDF(str(output_path), masterpath=str(generated_path), readonly=False) as cdf:
            cdf.attrs["Logical_file_id"] = output_path.stem



