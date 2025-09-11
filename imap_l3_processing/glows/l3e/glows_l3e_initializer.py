from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imap_data_access
from imap_data_access import ProcessingInputCollection, AncillaryInput, ScienceInput, ScienceFilePath, SPICEInput, \
    RepointInput

from imap_l3_processing.glows.l3bc.utils import get_pointing_date_range
from imap_l3_processing.glows.l3d.models import GlowsL3DProcessorOutput
from imap_l3_processing.glows.l3d.utils import get_most_recently_uploaded_ancillary
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_utils import find_first_updated_cr, determine_l3e_files_to_produce, \
    GlowsL3eRepointings


@dataclass
class GlowsL3EInitializerOutput:
    dependencies: GlowsL3EDependencies
    repointings: GlowsL3eRepointings


class GlowsL3EInitializer:
    @staticmethod
    def get_repointings_to_process(l3d_output: GlowsL3DProcessorOutput, previous_l3d: Optional[Path], repointing_file_path: Path) -> GlowsL3EInitializerOutput:
        ionization_files = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='ionization-files'))
        pipeline_settings_l3bcde = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='pipeline-settings-l3bcde'))
        energy_grid_lo = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-lo'))
        tess_xyz_8 = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='tess-xyz-8'))
        elongation_data = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='lo', descriptor='elongation-data'))
        energy_grid_hi = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-hi'))
        energy_grid_ultra = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-ultra'))
        tess_ang_16 = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='tess-ang-16'))

        processing_input_collection = ProcessingInputCollection(
            ScienceInput(l3d_output.l3d_cdf_file_path.name),
            *[AncillaryInput(file.name) for file in l3d_output.l3d_text_file_paths],
            AncillaryInput(str(ionization_files["file_path"])),
            AncillaryInput(str(pipeline_settings_l3bcde["file_path"])),
            AncillaryInput(str(energy_grid_lo["file_path"])),
            AncillaryInput(str(tess_xyz_8["file_path"])),
            AncillaryInput(str(elongation_data["file_path"])),
            AncillaryInput(str(energy_grid_hi["file_path"])),
            AncillaryInput(str(energy_grid_ultra["file_path"])),
            AncillaryInput(str(tess_ang_16["file_path"])),
            RepointInput(str(repointing_file_path))
        )

        l3e_deps = GlowsL3EDependencies.fetch_dependencies(processing_input_collection)
        l3e_deps.copy_dependencies()

        if previous_l3d is not None:
            first_cr = find_first_updated_cr(l3d_output.l3d_cdf_file_path, previous_l3d)
        else:
            first_cr = l3e_deps.pipeline_settings['start_cr']
        last_cr = ScienceFilePath(l3d_output.l3d_cdf_file_path).cr

        glows_repointings = determine_l3e_files_to_produce(first_cr, last_cr, repointing_file_path)

        if len(glows_repointings.repointing_numbers) > 0:
            earliest_repointing_start, _ = get_pointing_date_range(min(glows_repointings.repointing_numbers))
            _, latest_repointing_end = get_pointing_date_range(max(glows_repointings.repointing_numbers))

            l3e_deps.furnish_spice_dependencies(start_date=earliest_repointing_start, end_date=latest_repointing_end)

        return GlowsL3EInitializerOutput(
            dependencies=l3e_deps,
            repointings=glows_repointings
        )

