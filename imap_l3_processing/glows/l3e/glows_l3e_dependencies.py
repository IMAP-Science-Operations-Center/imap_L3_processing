import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from imap_data_access.processing_input import ProcessingInputCollection, RepointInput

from imap_l3_processing.utils import \
    download_dependency_from_path


@dataclass
class GlowsL3EDependencies:
    energy_grid_lo: Path | None
    energy_grid_hi: Path | None
    energy_grid_ultra: Path | None
    tess_xyz_8: Path | None
    tess_ang16: Path | None
    lya_series: Path
    solar_uv_anisotropy: Path
    speed_3d_sw: Path
    density_3d_sw: Path
    phion_hydrogen: Path
    sw_eqtr_electrons: Path
    ionization_files: Path
    pipeline_settings: dict
    elongation: dict
    repointing_file: Path

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection, descriptor: str):
        solar_hist_dependency = dependencies.get_file_paths(source='glows', descriptor='solar-hist')

        lya_series_dependency = dependencies.get_file_paths(source='glows', descriptor='lya-series')
        solar_uv_anisotropy_dependency = dependencies.get_file_paths(source='glows', descriptor='solar-uv-anisotropy')
        speed_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='speed-3d')
        density_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='density-3d')
        phion_hydrogen_dependency = dependencies.get_file_paths(source='glows', descriptor='phion-hydrogen')
        sw_eqtr_electrons_dependency = dependencies.get_file_paths(source='glows', descriptor='sw-eqtr-electrons')
        ionization_files_dependency = dependencies.get_file_paths(source='glows', descriptor='ionization-files')
        pipeline_settings_dependency = dependencies.get_file_paths(source='glows',
                                                                   descriptor='pipeline-settings-l3bcde')

        lya_series_path = download_dependency_from_path(str(lya_series_dependency[0]))
        solar_uv_anisotropy_path = download_dependency_from_path(str(solar_uv_anisotropy_dependency[0]))
        speed_3d_path = download_dependency_from_path(str(speed_3d_dependency[0]))
        density_3d_path = download_dependency_from_path(str(density_3d_dependency[0]))
        phion_hydrogen_path = download_dependency_from_path(str(phion_hydrogen_dependency[0]))
        sw_eqtr_electrons_path = download_dependency_from_path(str(sw_eqtr_electrons_dependency[0]))
        ionization_files_path = download_dependency_from_path(str(ionization_files_dependency[0]))
        pipeline_settings_path = download_dependency_from_path(str(pipeline_settings_dependency[0]))

        energy_grid_lo_path = None
        energy_grid_hi_path = None
        energy_grid_ultra_path = None
        tess_xyz_path = None
        tess_ang_path = None
        elongation_data = None

        if descriptor == "survival-probability-lo":
            energy_grid_lo_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-lo')
            tess_xyz_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-xyz-8')
            elongation_dependency = dependencies.get_file_paths(source='lo', descriptor='elongation-data')
            energy_grid_lo_path = download_dependency_from_path(str(energy_grid_lo_dependency[0]))
            tess_xyz_path = download_dependency_from_path(str(tess_xyz_dependency[0]))
            elongation_path = download_dependency_from_path(str(elongation_dependency[0]))
            with open(elongation_path) as f:
                elongation_data = {}
                lines = [line.rstrip('\n') for line in f.readlines()]
                for line in lines:
                    repointing, elongation = line.split('\t')
                    elongation_data[repointing] = int(elongation)
        elif descriptor == "survival-probability-hi-45" or descriptor == "survival-probability-hi-90":
            energy_grid_hi_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-hi')
            energy_grid_hi_path = download_dependency_from_path(str(energy_grid_hi_dependency[0]))
        elif descriptor == "survival-probability-ul":
            energy_grid_ultra_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-ultra')
            tess_ang_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-ang-16')
            energy_grid_ultra_path = download_dependency_from_path(str(energy_grid_ultra_dependency[0]))
            tess_ang_path = download_dependency_from_path(str(tess_ang_dependency[0]))

        with open(pipeline_settings_path) as f:
            pipeline_settings = json.load(f)

        cr_number = int(str(solar_hist_dependency).split('_')[-2][-5:])

        repoint_file_dependency = dependencies.get_file_paths(data_type=RepointInput.data_type)
        repoint_file_path = download_dependency_from_path(str(repoint_file_dependency[0]))

        return cls(
            energy_grid_lo_path,
            energy_grid_hi_path,
            energy_grid_ultra_path,
            tess_xyz_path,
            tess_ang_path,
            lya_series_path,
            solar_uv_anisotropy_path,
            speed_3d_path,
            density_3d_path,
            phion_hydrogen_path,
            sw_eqtr_electrons_path,
            ionization_files_path,
            pipeline_settings,
            elongation_data,
            repoint_file_path
        ), cr_number

    def rename_dependencies(self):
        if self.energy_grid_lo is not None:
            shutil.move(self.energy_grid_lo, self.pipeline_settings['executable_dependency_paths']['energy-grid-lo'])
        if self.energy_grid_hi is not None:
            shutil.move(self.energy_grid_hi, self.pipeline_settings['executable_dependency_paths']['energy-grid-hi'])
        if self.energy_grid_ultra is not None:
            shutil.move(self.energy_grid_ultra,
                        self.pipeline_settings['executable_dependency_paths']['energy-grid-ultra'])
        if self.tess_xyz_8 is not None:
            shutil.move(self.tess_xyz_8, self.pipeline_settings['executable_dependency_paths']['tess-xyz-8'])
        if self.tess_ang16 is not None:
            shutil.move(self.tess_ang16, self.pipeline_settings['executable_dependency_paths']['tess-ang-16'])

        shutil.move(self.lya_series, self.pipeline_settings['executable_dependency_paths']['lya-series'])
        shutil.move(self.solar_uv_anisotropy,
                    self.pipeline_settings['executable_dependency_paths']['solar-uv-anisotropy'])
        shutil.move(self.speed_3d_sw, self.pipeline_settings['executable_dependency_paths']['speed-3d'])
        shutil.move(self.density_3d_sw, self.pipeline_settings['executable_dependency_paths']['density-3d'])
        shutil.move(self.phion_hydrogen, self.pipeline_settings['executable_dependency_paths']['phion-hydrogen'])
        shutil.move(self.sw_eqtr_electrons, self.pipeline_settings['executable_dependency_paths']['sw-eqtr-electrons'])
        shutil.move(self.ionization_files, self.pipeline_settings['executable_dependency_paths']['ionization-files'])
