from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imap_data_access
import numpy as np
from imap_data_access import (
    ProcessingInputCollection,
    AncillaryInput,
    ScienceInput,
    RepointInput,
)
from imap_data_access.file_validation import Version
from imap_processing.spice.repoint import set_global_repoint_table_paths, get_repoint_data
from spacepy.pycdf import CDF

from imap_l3_processing.glows.descriptors import GLOWS_L3E_DESCRIPTORS, GLOWS_L3E_HI_90_DESCRIPTOR, \
    GLOWS_L3E_HI_45_DESCRIPTOR, GLOWS_L3E_LO_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR, GLOWS_L3E_ULTRA_HF_DESCRIPTOR
from imap_l3_processing.glows.l3bc.utils import get_pointing_date_range
from imap_l3_processing.glows.l3d.models import GlowsL3DProcessorOutput
from imap_l3_processing.glows.l3d.utils import get_most_recently_uploaded_ancillary
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_utils import (
    GlowsL3eVersionsForRepointings, get_repoint_numbers_within_cr_window, )
from imap_l3_processing.glows.l3e.reprocess_info import ReprocessInfo
from imap_l3_processing.models import VersionMap
from imap_l3_processing.utils import FurnishMetakernelOutput

logger = logging.getLogger(__name__)

@dataclass
class GlowsL3EInitializerOutput:
    dependencies: GlowsL3EDependencies
    repointings: GlowsL3eVersionsForRepointings
    l3d_cdf_path: Path
    metakernel_with_predict_ephem: FurnishMetakernelOutput
    metakernel_without_predict_ephem: FurnishMetakernelOutput


class GlowsL3EInitializer:
    @staticmethod
    def get_repointings_to_process(
        l3d_output: GlowsL3DProcessorOutput,
        previous_l3d: Optional[str],
        repointing_file_path: Path,
        version_map: VersionMap,
        reprocess_info: ReprocessInfo,
    ) -> Optional[GlowsL3EInitializerOutput]:
        pipeline_settings_l3bcde = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='pipeline-settings-l3bcde'))
        energy_grid_lo = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='energy-grid-lo'))
        tess_xyz_8 = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='tess-xyz-8'))
        energy_grid_hi = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='energy-grid-hi'))
        energy_grid_ultra = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='energy-grid-ultra'))
        tess_ang_16 = get_most_recently_uploaded_ancillary(imap_data_access.query(table='ancillary', instrument='glows', descriptor='tess-ang-16'))

        processing_input_collection = ProcessingInputCollection(
            ScienceInput(l3d_output.l3d_cdf_file_path.name),
            *[AncillaryInput(file.name) for file in l3d_output.l3d_text_file_paths],
            AncillaryInput(str(pipeline_settings_l3bcde["file_path"])),
            AncillaryInput(str(energy_grid_lo["file_path"])),
            AncillaryInput(str(tess_xyz_8["file_path"])),
            AncillaryInput(str(energy_grid_hi["file_path"])),
            AncillaryInput(str(energy_grid_ultra["file_path"])),
            AncillaryInput(str(tess_ang_16["file_path"])),
            RepointInput(str(repointing_file_path))
        )

        l3e_deps = GlowsL3EDependencies.fetch_dependencies(processing_input_collection)
        l3e_deps.copy_dependencies()

        first_cr = l3e_deps.pipeline_settings["start_cr"]

        first_updated_cr = first_cr
        if previous_l3d is not None:
            first_updated_cr = find_first_updated_cr(l3d_output.l3d_cdf_file_path, previous_l3d)
            if first_updated_cr is not None:
                first_updated_cr -= 1

        last_cr = l3d_output.last_processed_cr
        files_to_produce = identify_versions_for_l3e_output_files(
            first_cr,
            last_cr,
            first_updated_cr,
            repointing_file_path,
            version_map,
            reprocess_info,
        )

        if len(files_to_produce.repointing_numbers) == 0:
            return None

        earliest_repointing_start, _ = get_pointing_date_range(
            min(files_to_produce.repointing_numbers)
        )
        _, latest_repointing_end = get_pointing_date_range(
            max(files_to_produce.repointing_numbers)
        )

        furnished_metakernels = GlowsL3EDependencies.collect_spice_dependencies(
            start_date=earliest_repointing_start, end_date=latest_repointing_end
        )

        return GlowsL3EInitializerOutput(
            dependencies=l3e_deps,
            repointings=files_to_produce,
            l3d_cdf_path=l3d_output.l3d_cdf_file_path,
            metakernel_with_predict_ephem=furnished_metakernels[0],
            metakernel_without_predict_ephem=furnished_metakernels[1],
        )

def query_existing_l3es_versions(descriptor: str) -> dict[int, Version]:
    existing_l3es_for_descriptor = {}
    l3e_files = imap_data_access.query(instrument='glows', data_level='l3e', version="latest", descriptor=descriptor)
    for l3e in l3e_files:
        if 'major_version' in l3e.keys() and 'minor_version' in l3e.keys():
            existing_l3es_for_descriptor[int(l3e['repointing'])] = Version(l3e["major_version"], l3e["minor_version"])
        elif 'version' in l3e.keys():
            existing_l3es_for_descriptor[int(l3e['repointing'])] = Version.from_version(l3e['version'])
        else:
            continue
    return existing_l3es_for_descriptor

def identify_versions_for_l3e_output_files(start_cr_of_mission: int, end_cr_of_mission: int, first_updated_cr_from_l3d: Optional[int],
                                           repointing_path: Path, version_map: VersionMap, reprocess_info: ReprocessInfo) -> GlowsL3eVersionsForRepointings:

    set_global_repoint_table_paths([repointing_path])
    repointing_data = get_repoint_data()

    all_pointing_numbers = get_repoint_numbers_within_cr_window(start_cr_of_mission, end_cr_of_mission, repointing_data)
    pointing_numbers_updated_by_l3d = get_repoint_numbers_within_cr_window(first_updated_cr_from_l3d, end_cr_of_mission, repointing_data)

    updated_pointings_per_instruments = {}
    updated_pointing_numbers = {}

    for descriptor in GLOWS_L3E_DESCRIPTORS:
        new_major_version = version_map.lookup(descriptor).major

        repointings_to_force_processing = reprocess_info.get_repoints_for_descriptor(
            descriptor, repointing_data
        )

        existing_file_versions = query_existing_l3es_versions(descriptor)
        out_of_date_repointings = find_out_of_date_l3es(existing_file_versions, all_pointing_numbers, new_major_version)

        repointings_to_process = pointing_numbers_updated_by_l3d | repointings_to_force_processing | out_of_date_repointings

        new_file_versions = {}
        for pointing_number in sorted(repointings_to_process):
            if previous_version := existing_file_versions.get(pointing_number):
                new_version = Version(new_major_version, previous_version.minor + 1)
            else:
                new_version = Version(new_major_version, 1)
            new_file_versions[pointing_number] = new_version

        updated_pointings_per_instruments[descriptor] = new_file_versions
        updated_pointing_numbers = updated_pointing_numbers | new_file_versions.keys()


    return GlowsL3eVersionsForRepointings(list(updated_pointing_numbers),
                                          updated_pointings_per_instruments[GLOWS_L3E_HI_90_DESCRIPTOR],
                                          updated_pointings_per_instruments[GLOWS_L3E_HI_45_DESCRIPTOR],
                                          updated_pointings_per_instruments[GLOWS_L3E_LO_DESCRIPTOR],
                                          updated_pointings_per_instruments[GLOWS_L3E_ULTRA_SF_DESCRIPTOR],
                                          updated_pointings_per_instruments[GLOWS_L3E_ULTRA_HF_DESCRIPTOR],
                                          )

def find_out_of_date_l3es(existing_l3es: dict[int, Version], repointings_to_check: set[int], current_major_version: int) -> set[int]:
    out_of_date_l3es = set()
    for repointing in repointings_to_check:
        if existing_l3e_version := existing_l3es.get(repointing):
            if existing_l3e_version.major < current_major_version:
                out_of_date_l3es.add(repointing)
        else:
            out_of_date_l3es.add(repointing)
    return out_of_date_l3es


def find_first_updated_cr(new_l3d: Path, old_l3d: str) -> Optional[int]:
    downloaded_old_l3d = imap_data_access.download(old_l3d)

    old_l3d_cdf = CDF(str(downloaded_old_l3d))
    new_l3d_cdf = CDF(str(new_l3d))

    for i, cr in enumerate(old_l3d_cdf['cr_grid'][...]):
        lya_matches = np.isclose(old_l3d_cdf['lyman_alpha'][i], new_l3d_cdf['lyman_alpha'][i])
        phion_matches = np.isclose(old_l3d_cdf['phion'][i], new_l3d_cdf['phion'][i])
        plasma_speed_flag_matches = np.isclose(old_l3d_cdf['plasma_speed_flag'][i], new_l3d_cdf['plasma_speed_flag'][i])
        proton_density_flag_matches = np.isclose(old_l3d_cdf['proton_density_flag'][i], new_l3d_cdf['proton_density_flag'][i])
        uv_anisotropy_flag_matches = np.isclose(old_l3d_cdf['uv_anisotropy_flag'][i], new_l3d_cdf['uv_anisotropy_flag'][i])
        glows_flags_matches = np.isclose(old_l3d_cdf['glows_flags'][i], new_l3d_cdf['glows_flags'][i])

        plasma_speed_matches = np.all(np.isclose(old_l3d_cdf['plasma_speed'][i], new_l3d_cdf['plasma_speed'][i]))
        proton_density_matches = np.all(np.isclose(old_l3d_cdf['proton_density'][i], new_l3d_cdf['proton_density'][i]))
        uv_anisotropy_matches = np.all(np.isclose(old_l3d_cdf['uv_anisotropy'][i], new_l3d_cdf['uv_anisotropy'][i]))

        if np.any(np.logical_not([lya_matches, phion_matches, plasma_speed_matches, plasma_speed_flag_matches, proton_density_matches, proton_density_flag_matches, uv_anisotropy_matches, uv_anisotropy_flag_matches, glows_flags_matches])):
            return int(cr)

    if old_l3d_cdf['cr_grid'].shape != new_l3d_cdf['cr_grid'].shape:
        return int(old_l3d_cdf['cr_grid'][-1]) + 1

    return None
