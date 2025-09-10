import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, RepointInput

from imap_l3_processing.utils import furnish_spice_metakernel, SpiceKernelTypes

GLOWS_L3E_REQUIRED_SPICE_KERNELS: list[SpiceKernelTypes] = [
    SpiceKernelTypes.ScienceFrames, SpiceKernelTypes.EphemerisReconstructed, SpiceKernelTypes.AttitudeHistory,
    SpiceKernelTypes.PointingAttitude, SpiceKernelTypes.PlanetaryEphemeris, SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.SpacecraftClock
]

@dataclass
class GlowsL3EDependencies:
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
    ionization_files: Path
    pipeline_settings: dict
    pipeline_settings_file: Path
    elongation: dict
    elongation_file: Path
    repointing_file: Path
    kernels: list[str] = field(default_factory=list)

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        lya_series_dependency = dependencies.get_file_paths(source='glows', descriptor='lya')
        solar_uv_anisotropy_dependency = dependencies.get_file_paths(source='glows', descriptor='uv-anis')
        speed_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='speed')
        density_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='p-dens')
        phion_hydrogen_dependency = dependencies.get_file_paths(source='glows', descriptor='phion')
        sw_eqtr_electrons_dependency = dependencies.get_file_paths(source='glows', descriptor='e-dens')

        ionization_files_dependency = dependencies.get_file_paths(source='glows', descriptor='ionization-files')
        pipeline_settings_dependency = dependencies.get_file_paths(source='glows', descriptor='pipeline-settings-l3bcde')

        lya_series_path = imap_data_access.download(lya_series_dependency[0])
        solar_uv_anisotropy_path = imap_data_access.download(solar_uv_anisotropy_dependency[0])
        speed_3d_path = imap_data_access.download(speed_3d_dependency[0])
        density_3d_path = imap_data_access.download(density_3d_dependency[0])
        phion_hydrogen_path = imap_data_access.download(phion_hydrogen_dependency[0])
        sw_eqtr_electrons_path = imap_data_access.download(sw_eqtr_electrons_dependency[0])
        ionization_files_path = imap_data_access.download(ionization_files_dependency[0])
        pipeline_settings_path = imap_data_access.download(pipeline_settings_dependency[0])

        energy_grid_lo_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-lo')
        tess_xyz_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-xyz-8')
        elongation_dependency = dependencies.get_file_paths(source='lo', descriptor='elongation-data')
        energy_grid_lo_path = imap_data_access.download(energy_grid_lo_dependency[0])
        tess_xyz_path = imap_data_access.download(tess_xyz_dependency[0])
        elongation_path = imap_data_access.download(elongation_dependency[0])
        with open(elongation_path) as f:
            elongation_data = {}
            lines = [line.rstrip('\n') for line in f.readlines()]
            for line in lines:
                repointing, elongation = line.split('\t')
                elongation_data[repointing] = int(elongation)

        energy_grid_hi_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-hi')
        energy_grid_hi_path = imap_data_access.download(energy_grid_hi_dependency[0])

        energy_grid_ultra_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-ultra')
        tess_ang_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-ang-16')
        energy_grid_ultra_path = imap_data_access.download(energy_grid_ultra_dependency[0])
        tess_ang_path = imap_data_access.download(tess_ang_dependency[0])

        with open(pipeline_settings_path) as f:
            pipeline_settings = json.load(f)

        repoint_file_dependency = dependencies.get_file_paths(data_type=RepointInput.data_type)
        repoint_file_path = imap_data_access.download(repoint_file_dependency[0])

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
            pipeline_settings_path,
            elongation_data,
            elongation_path,
            repoint_file_path
        )

    def furnish_spice_dependencies(self, start_date: datetime, end_date: datetime):
        kernels = furnish_spice_metakernel(start_date=start_date, end_date=end_date, kernel_types=GLOWS_L3E_REQUIRED_SPICE_KERNELS)

        self.kernels.extend([kernel_path.name for kernel_path in kernels.spice_kernel_paths])

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

    def get_hi_parents(self):
        return [
            self.energy_grid_hi.name,
            self.lya_series.name,
            self.solar_uv_anisotropy.name,
            self.speed_3d_sw.name,
            self.density_3d_sw.name,
            self.phion_hydrogen.name,
            self.sw_eqtr_electrons.name,
            self.ionization_files.name,
            self.pipeline_settings_file.name,
            self.repointing_file.name,
            *self.kernels
        ]

    def get_lo_parents(self):
        return [
            self.energy_grid_lo.name,
            self.tess_xyz_8.name,
            self.lya_series.name,
            self.solar_uv_anisotropy.name,
            self.speed_3d_sw.name,
            self.density_3d_sw.name,
            self.phion_hydrogen.name,
            self.sw_eqtr_electrons.name,
            self.ionization_files.name,
            self.pipeline_settings_file.name,
            self.repointing_file.name,
            self.elongation_file.name,
            *self.kernels
        ]

    def get_ul_parents(self):
        return [
            self.energy_grid_ultra.name,
            self.tess_ang16.name,
            self.lya_series.name,
            self.solar_uv_anisotropy.name,
            self.speed_3d_sw.name,
            self.density_3d_sw.name,
            self.phion_hydrogen.name,
            self.sw_eqtr_electrons.name,
            self.ionization_files.name,
            self.pipeline_settings_file.name,
            self.repointing_file.name,
            *self.kernels
        ]
