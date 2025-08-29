from typing import Optional

import imap_data_access

from imap_l3_processing.glows.l3d.utils import get_most_recently_uploaded_ancillary
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies


class GlowsL3EInitializerOutput:
    dependencies: list[GlowsL3EDependencies]

    # l3e_ultra_dependency: Optional[GlowsL3EDependencies]
    # l3e_hi_45_dependency: Optional[GlowsL3EDependencies]
    # l3e_hi_90_dependency: Optional[GlowsL3EDependencies]
    # l3e_lo_dependency: Optional[GlowsL3EDependencies]

    repointing_and_version_to_process: list[tuple[int, int]]


class GlowsL3EInitializer:
    @staticmethod
    def determine_l3e_files_to_produce():

        ionization_files = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='ionization-files'))

        pipeline_settings_l3bcde = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='pipeline-settings-l3bcde'))
        energy_grid_lo = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-lo'))
        tess_xyz_8 = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='tess-xyz-8'))
        elongation_data = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='lo', descriptor='elongation-data'))
        energy_grid_hi = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-hi'))
        energy_grid_ultra = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='energy-grid-ultra'))
        tess_ang_16 = get_most_recently_uploaded_ancillary(imap_data_access.query(instrument='glows', descriptor='tess-ang-16'))