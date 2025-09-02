from pathlib import Path
from typing import Optional

import imap_data_access
from imap_data_access import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.glows.l3bc.models import ExternalDependencies
from imap_l3_processing.glows.l3bc.utils import read_cdf_parents
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.utils import query_for_most_recent_l3d


class GlowsL3DInitializer:

    @staticmethod
    def should_process_l3d(external_deps: ExternalDependencies, l3bs: list[str], l3cs: list[str]) -> Optional[
        tuple[int, GlowsL3DDependencies, Optional[Path]]]:
        most_recent_l3d = query_for_most_recent_l3d("solar-hist")

        # @formatter:off
        [plasma_speed_2010a] = imap_data_access.query(table='ancillary', instrument='glows', descriptor='plasma-speed-2010a', version='latest')
        [proton_density_2010a] = imap_data_access.query(table='ancillary', instrument='glows', descriptor='proton-density-2010a', version='latest')
        [uv_anisotropy_2010a] = imap_data_access.query(table='ancillary', instrument='glows', descriptor='uv-anisotropy-2010a', version='latest')
        [photoion_2010a] = imap_data_access.query(table='ancillary', instrument='glows', descriptor='photoion-2010a', version='latest')
        [lya_2010a] = imap_data_access.query(table='ancillary', instrument='glows', descriptor='lya-2010a', version='latest')
        [electron_density_2010a] = imap_data_access.query(table='ancillary', instrument='glows', descriptor='electron-density-2010a', version='latest')
        [pipeline_settings_l3bcde] = imap_data_access.query(table='ancillary', instrument='glows', descriptor='pipeline-settings-l3bcde', version='latest')
        # @formatter:on

        processing_input_collection = ProcessingInputCollection(
            *[ScienceInput(science_file) for science_file in l3bs + l3cs],
            AncillaryInput(plasma_speed_2010a["file_path"]),
            AncillaryInput(proton_density_2010a["file_path"]),
            AncillaryInput(uv_anisotropy_2010a["file_path"]),
            AncillaryInput(photoion_2010a["file_path"]),
            AncillaryInput(lya_2010a["file_path"]),
            AncillaryInput(electron_density_2010a["file_path"]),
            AncillaryInput(pipeline_settings_l3bcde["file_path"]),
        )

        updated_input_files = {
            *l3bs,
            *l3cs,
            Path(plasma_speed_2010a['file_path']).name,
            Path(proton_density_2010a['file_path']).name,
            Path(uv_anisotropy_2010a['file_path']).name,
            Path(photoion_2010a['file_path']).name,
            Path(lya_2010a['file_path']).name,
            Path(electron_density_2010a['file_path']).name,
            Path(pipeline_settings_l3bcde['file_path']).name,
        }

        if most_recent_l3d is not None:
            l3d_parents = read_cdf_parents(most_recent_l3d["file_path"])
            old_l3d = most_recent_l3d["file_path"]
            if updated_input_files.issubset(l3d_parents):
                return None
            version_to_generate = int(most_recent_l3d['version'][1:]) + 1
        else:
            old_l3d = None
            version_to_generate = 1

        return (version_to_generate, GlowsL3DDependencies.fetch_dependencies(processing_input_collection, external_deps),
                old_l3d)
