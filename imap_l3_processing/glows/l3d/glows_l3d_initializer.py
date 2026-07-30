import logging
from pathlib import Path
from typing import Optional

import imap_data_access
from imap_data_access import ProcessingInputCollection, ScienceInput, AncillaryInput, ScienceFilePath
from imap_data_access.file_validation import Version

from imap_l3_processing.glows.descriptors import PLASMA_SPEED_DESCRIPTOR, PROTON_DENSITY_DESCRIPTOR, \
    UV_ANISOTROPY_DESCRIPTOR, PHOTOION_DESCRIPTOR, LYA_DESCRIPTOR, ELECTRON_DENSITY_DESCRIPTOR, \
    PIPELINE_SETTINGS_L3BCDE_DESCRIPTOR, GLOWS_L3D_DESCRIPTOR
from imap_l3_processing.glows.l3bc.models import ExternalDependencies, read_pipeline_settings
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.utils import query_for_most_recent_l3d
from imap_l3_processing.utils import read_cdf_parents, get_version_from_query_result

logger = logging.getLogger(__name__)


class GlowsL3DInitializer:

    @staticmethod
    def should_process_l3d(external_deps: ExternalDependencies, l3bs: list[str], l3cs: list[str], major_version: int|None) -> Optional[
        tuple[Version, GlowsL3DDependencies, Optional[str]]]:
        if len(l3bs) == 0 and len(l3cs) == 0:
            logger.info("Found no L3b and L3c files!")
            return None

        most_recent_l3d = query_for_most_recent_l3d(GLOWS_L3D_DESCRIPTOR)

        [plasma_speed_2026a] = imap_data_access.query(
            table='ancillary',
            instrument='glows',
            descriptor=PLASMA_SPEED_DESCRIPTOR,
            version='latest'
        )
        [proton_density_2026a] = imap_data_access.query(
            table='ancillary',
            instrument='glows',
            descriptor=PROTON_DENSITY_DESCRIPTOR,
            version='latest'
        )
        [uv_anisotropy_2026a] = imap_data_access.query(
            table='ancillary',
            instrument='glows',
            descriptor=UV_ANISOTROPY_DESCRIPTOR,
            version='latest')
        [photoion_2026a] = imap_data_access.query(
            table='ancillary',
            instrument='glows',
            descriptor=PHOTOION_DESCRIPTOR,
            version='latest'
        )
        [lya_2026a] = imap_data_access.query(
            table='ancillary',
            instrument='glows',
            descriptor=LYA_DESCRIPTOR,
            version='latest'
        )
        [electron_density_2026a] = imap_data_access.query(
            table='ancillary',
            instrument='glows',
            descriptor=ELECTRON_DENSITY_DESCRIPTOR,
            version='latest'
        )
        [pipeline_settings_l3bcde] = imap_data_access.query(
            table='ancillary',
            instrument='glows',
            descriptor=PIPELINE_SETTINGS_L3BCDE_DESCRIPTOR,
            version='latest'
        )

        pipeline_settings_filename = Path(pipeline_settings_l3bcde["file_path"]).name
        pipeline_settings = read_pipeline_settings(imap_data_access.download(pipeline_settings_filename))

        l3bs = [l3b for l3b in l3bs if ScienceFilePath(l3b).cr >= pipeline_settings["start_cr"]]
        l3cs = [l3c for l3c in l3cs if ScienceFilePath(l3c).cr >= pipeline_settings["start_cr"]]

        if len(l3bs) == 0 and len(l3cs) == 0:
            logger.info("Found no L3b and L3c files after start CR!")
            return None

        processing_input_collection = ProcessingInputCollection(
            *[ScienceInput(science_file) for science_file in l3bs + l3cs],
            AncillaryInput(plasma_speed_2026a["file_path"]),
            AncillaryInput(proton_density_2026a["file_path"]),
            AncillaryInput(uv_anisotropy_2026a["file_path"]),
            AncillaryInput(photoion_2026a["file_path"]),
            AncillaryInput(lya_2026a["file_path"]),
            AncillaryInput(electron_density_2026a["file_path"]),
            AncillaryInput(pipeline_settings_l3bcde["file_path"]),
        )

        updated_input_files = {
            *l3bs,
            *l3cs,
            Path(plasma_speed_2026a['file_path']).name,
            Path(proton_density_2026a['file_path']).name,
            Path(uv_anisotropy_2026a['file_path']).name,
            Path(photoion_2026a['file_path']).name,
            Path(lya_2026a['file_path']).name,
            Path(electron_density_2026a['file_path']).name,
        }

        if most_recent_l3d is not None:
            l3d_parents = read_cdf_parents(Path(most_recent_l3d["file_path"]).name)
            old_l3d = Path(most_recent_l3d["file_path"]).name

            logger.info(
                f"Old L3d parents: {l3d_parents}, new L3d deps: {updated_input_files}"
            )

            most_recent_l3d_version = get_version_from_query_result(most_recent_l3d)
            same_major_version = most_recent_l3d_version.major == major_version
            if same_major_version and updated_input_files.issubset(l3d_parents):
                return None
            minor_version_to_generate = most_recent_l3d_version.minor + 1
        else:
            old_l3d = None
            minor_version_to_generate = 1
        version_to_generate = Version(major_version, minor_version_to_generate)

        return (version_to_generate,
                GlowsL3DDependencies.fetch_dependencies(processing_input_collection, external_deps),
                old_l3d)
