from dataclasses import dataclass
from pathlib import Path
from shutil import move

from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.glows.descriptors import GLOWS_L3D_DESCRIPTOR
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency, download_dependency_with_repointing, \
    download_dependency_from_path


@dataclass
class GlowsL3EDependencies:
    l3d_data: Path
    energy_grid_lo: Path
    energy_grid_hi: Path
    energy_grid_ultra: Path
    tess_xyz_8: Path
    tess_ang16: Path
    lya_series: Path
    solar_uv_anisotropy: Path
    speed_3d_sw: Path
    density_3d_sw: Path
    phion_hydrogen: Path
    sw_eqtr_electrons: Path
    pipeline_settings: dict

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        solar_hist_dependency = dependencies.get_file_paths(source='glows', descriptor='solar-hist')
        energy_grid_lo_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-lo')
        energy_grid_hi_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-hi')
        energy_grid_ultra_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-ultra')
        tess_xyz_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-xyz-8')
        tess_ang_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-ang-16')
        lya_series_dependency = dependencies.get_file_paths(source='glows', descriptor='lya-series')
        solar_uv_anisotropy_dependency = dependencies.get_file_paths(source='glows', descriptor='solar-uv-anistropy')
        speed_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='speed-3d')
        density_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='density-3d')
        phion_hydrogen_dependency = dependencies.get_file_paths(source='glows', descriptor='phion-hydrogen')
        sw_eqtr_electrons_dependency = dependencies.get_file_paths(source='glows', descriptor='sw-eqtr-electrons')

        solar_hist_path = download_dependency_from_path(str(solar_hist_dependency[0]))
        energy_grid_lo_path = download_dependency_from_path(str(energy_grid_lo_dependency[0]))
        energy_grid_hi_path = download_dependency_from_path(str(energy_grid_hi_dependency[0]))
        energy_grid_ultra_path = download_dependency_from_path(str(energy_grid_ultra_dependency[0]))
        tess_xyz_path = download_dependency_from_path(str(tess_xyz_dependency[0]))
        tess_ang_path = download_dependency_from_path(str(tess_ang_dependency[0]))
        lya_series_path = download_dependency_from_path(str(lya_series_dependency[0]))
        solar_uv_anisotropy__path = download_dependency_from_path(str(solar_uv_anisotropy_dependency[0]))
        speed_3d_path = download_dependency_from_path(str(speed_3d_dependency[0]))
        density_3d_path = download_dependency_from_path(str(density_3d_dependency[0]))
        phion_hydrogen_path = download_dependency_from_path(str(phion_hydrogen_dependency[0]))
        sw_eqtr_electrons_path = download_dependency_from_path(str(sw_eqtr_electrons_dependency[0]))

        repointing_number = int(str(solar_hist_path).split('_')[-2][-5:])

        return cls(
            solar_hist_path,
            energy_grid_lo_path,
            energy_grid_hi_path,
            energy_grid_ultra_path,
            tess_xyz_path,
            tess_ang_path,
            lya_series_path,
            solar_uv_anisotropy__path,
            speed_3d_path,
            density_3d_path,
            phion_hydrogen_path,
            sw_eqtr_electrons_path,
            {},
        ), repointing_number

    def rename_dependencies(self):
        move(self.energy_grid_lo, self.pipeline_settings['executable_dependency_paths']['energy-grid-lo'])
        move(self.energy_grid_hi, self.pipeline_settings['executable_dependency_paths']['energy-grid-hi'])
        move(self.energy_grid_ultra, self.pipeline_settings['executable_dependency_paths']['energy-grid-ultra'])
        move(self.tess_xyz_8, self.pipeline_settings['executable_dependency_paths']['tess-xyz-8'])
        move(self.tess_ang16, self.pipeline_settings['executable_dependency_paths']['tess-ang-16'])
        move(self.lya_series, self.pipeline_settings['executable_dependency_paths']['lya-series'])
        move(self.solar_uv_anisotropy, self.pipeline_settings['executable_dependency_paths']['solar-uv-anistropy'])
        move(self.speed_3d_sw, self.pipeline_settings['executable_dependency_paths']['speed-3d'])
        move(self.density_3d_sw, self.pipeline_settings['executable_dependency_paths']['density-3d'])
        move(self.phion_hydrogen, self.pipeline_settings['executable_dependency_paths']['phion-hydrogen'])
        move(self.sw_eqtr_electrons,
             self.pipeline_settings['executable_dependency_paths']['sw-eqtr-electrons'])
