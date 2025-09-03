from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imap_data_access
from imap_data_access import ProcessingInputCollection, AncillaryInput, ScienceInput, ScienceFilePath

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
    def get_repointings_to_process(updated_l3d: Path, previous_l3d: Optional[Path]) -> GlowsL3EInitializerOutput:
        ionization_files = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='ionization-files'))
        pipeline_settings_l3bcde = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='pipeline-settings-l3bcde'))
        energy_grid_lo = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-lo'))
        tess_xyz_8 = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='tess-xyz-8'))
        elongation_data = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='lo', descriptor='elongation-data'))
        energy_grid_hi = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-hi'))
        energy_grid_ultra = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-ultra'))
        tess_ang_16 = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='tess-ang-16'))

        processing_input_collection = ProcessingInputCollection(
            ScienceInput(updated_l3d.name),
            AncillaryInput(str(ionization_files)),
            AncillaryInput(str(pipeline_settings_l3bcde)),
            AncillaryInput(str(energy_grid_lo)),
            AncillaryInput(str(tess_xyz_8)),
            AncillaryInput(str(elongation_data)),
            AncillaryInput(str(energy_grid_hi)),
            AncillaryInput(str(energy_grid_ultra)),
            AncillaryInput(str(tess_ang_16)),
        )

        deps = GlowsL3EDependencies.fetch_dependencies(processing_input_collection)
        deps.rename_dependencies()

        if previous_l3d is not None:
            first_cr = find_first_updated_cr(updated_l3d, previous_l3d)
        else:
            first_cr = deps.pipeline_settings['start_cr']
        last_cr = ScienceFilePath(updated_l3d).cr

        glows_repointings = determine_l3e_files_to_produce(first_cr, last_cr, deps.repointing_file)

        return GlowsL3EInitializerOutput(
            dependencies=deps,
            repointings=glows_repointings
        )
