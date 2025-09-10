import unittest
from pathlib import Path
from unittest.mock import patch, Mock, call, sentinel

from imap_data_access import RepointInput

from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.utils import SpiceKernelTypes, FurnishMetakernelOutput
from tests.test_helpers import get_test_data_path


class TestGlowsL3EDependencies(unittest.TestCase):
    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.imap_data_access.download')
    def test_fetch_dependencies(self, mock_download):
        mock_processing_input_collection = Mock()

        mock_energy_grid_lo = Path('energy_grid_lo_path')
        mock_energy_grid_hi = Path('energy_grid_hi_sdc_path')
        mock_energy_grid_ultra = Path('energy_grid_ultra_sdc_path')
        mock_tess_xyz_8 = Path('tess_xyz_8_sdc_path')
        mock_lya_series = Path('lya_series_sdc_path')
        mock_solar_uv_anisotropy = Path('solar_uv_anisotropy_sdc_path')
        mock_speed_3d_sw = Path('speed_3d_sw_sdc_path')
        mock_density_3d_sw = Path('density_3d_sw_sdc_path')
        mock_phion_hydrogen = Path('phion_hydrogen_sdc_path')
        mock_sw_eqtr_electrons = Path('sw_eqtr_electrons_sdc_path')
        ionization_files = Path('ionization_files_path')
        mock_pipeline_settings = Path('l3bcde_pipeline_settings.json')
        mock_lo_elongation_file = Path('lo_elongation_data.dat')
        mock_tess_ang_16 = Path('tess_ang_16_path')
        mock_repoint_file = Path('repoint.csv')

        mock_processing_input_collection.get_file_paths.side_effect = [
            [mock_lya_series], [mock_solar_uv_anisotropy], [mock_speed_3d_sw], [mock_density_3d_sw],
            [mock_phion_hydrogen], [mock_sw_eqtr_electrons], [ionization_files], [mock_pipeline_settings],
            [mock_tess_xyz_8], [mock_energy_grid_lo], [mock_lo_elongation_file],
            [mock_energy_grid_hi],
            [mock_energy_grid_ultra], [mock_tess_ang_16],
            [mock_repoint_file]
        ]

        mock_lya_series_path = Mock()
        mock_solar_uv_anisotropy_path = Mock()
        mock_speed_3d_sw_path = Mock()
        mock_density_3d_sw_path = Mock()
        mock_phion_hydrogen_path = Mock()
        mock_sw_eqtr_electrons_path = Mock()
        fake_pipeline_settings_path = get_test_data_path(
            "glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json")
        mock_ionization_files_path = Mock()
        mock_energy_grid_lo_path = Mock()
        mock_tess_xyz_8_path = Mock()
        mock_lo_elongation_path = get_test_data_path('glows/imap_lo_elongation-data_20100101_v001.dat')
        mock_energy_grid_hi_path = Mock()
        mock_energy_grid_ultra_path = Mock()
        mock_tess_ang_16_path = Mock()
        mock_downloaded_repoint_file = Mock()

        mock_download.side_effect = [
            mock_lya_series_path, mock_solar_uv_anisotropy_path, mock_speed_3d_sw_path,
            mock_density_3d_sw_path, mock_phion_hydrogen_path, mock_sw_eqtr_electrons_path, mock_ionization_files_path,
            fake_pipeline_settings_path, mock_energy_grid_lo_path, mock_tess_xyz_8_path, mock_lo_elongation_path,
            mock_energy_grid_hi_path,
            mock_energy_grid_ultra_path, mock_tess_ang_16_path,
            mock_downloaded_repoint_file,
        ]

        actual_dependencies = GlowsL3EDependencies.fetch_dependencies(mock_processing_input_collection)

        self.assertEqual(15, mock_processing_input_collection.get_file_paths.call_count)

        mock_processing_input_collection.get_file_paths.assert_has_calls([
            call(source="glows", descriptor="lya"),
            call(source="glows", descriptor="uv-anis"),
            call(source="glows", descriptor="speed"),
            call(source="glows", descriptor="p-dens"),
            call(source="glows", descriptor="phion"),
            call(source="glows", descriptor="e-dens"),
            call(source="glows", descriptor="ionization-files"),
            call(source="glows", descriptor="pipeline-settings-l3bcde"),
            call(source="glows", descriptor="energy-grid-lo"),
            call(source="glows", descriptor="tess-xyz-8"),
            call(source="lo", descriptor="elongation-data"),
            call(source="glows", descriptor="energy-grid-hi"),
            call(source="glows", descriptor="energy-grid-ultra"),
            call(source="glows", descriptor="tess-ang-16"),
            call(data_type=RepointInput.data_type),
        ])

        mock_download.assert_has_calls([
            call(mock_lya_series),
            call(mock_solar_uv_anisotropy),
            call(mock_speed_3d_sw),
            call(mock_density_3d_sw),
            call(mock_phion_hydrogen),
            call(mock_sw_eqtr_electrons),
            call(ionization_files),
            call(mock_pipeline_settings),
            call(mock_tess_xyz_8),
            call(mock_energy_grid_lo),
            call(mock_lo_elongation_file),
            call(mock_energy_grid_hi),
            call(mock_energy_grid_ultra),
            call(mock_tess_ang_16),
            call(mock_repoint_file),
        ], any_order=False)

        expected_pipeline_settings = {"executable_dependency_paths": {
            "energy-grid-lo": "EnGridLo.dat",
            "energy-grid-hi": "EnGridHi.dat",
            "energy-grid-ultra": "EnGridUltra.dat",
            "tess-xyz-8": "tessXYZ8.dat",
            "tess-ang-16": "tessAng16.dat",
            "lya-series": "lyaSeriesV4_2021b.dat",
            "solar-uv-anisotropy": "solar_uv_anisotropy_NP.1.0_SP.1.0.dat",
            "speed-3d": "speed3D.v01.Legendre.2021b.dat",
            "density-3d": "density3D.v01.Legendre.2021b.dat",
            "phion-hydrogen": "phion_Hydrogen_T12F107_2021b.dat",
            "sw-eqtr-electrons": "swEqtrElectrons5_2021b.dat",
            "ionization-files": "ionization.files.dat",
        }}

        self.assertEqual(mock_lya_series_path, actual_dependencies.lya_series)
        self.assertEqual(mock_solar_uv_anisotropy_path, actual_dependencies.solar_uv_anisotropy)
        self.assertEqual(mock_speed_3d_sw_path, actual_dependencies.speed_3d_sw)
        self.assertEqual(mock_density_3d_sw_path, actual_dependencies.density_3d_sw)
        self.assertEqual(mock_phion_hydrogen_path, actual_dependencies.phion_hydrogen)
        self.assertEqual(mock_sw_eqtr_electrons_path, actual_dependencies.sw_eqtr_electrons)
        self.assertEqual(expected_pipeline_settings['executable_dependency_paths'],
                         actual_dependencies.pipeline_settings['executable_dependency_paths'])
        self.assertEqual(mock_ionization_files_path, actual_dependencies.ionization_files)
        self.assertEqual(mock_energy_grid_lo_path, actual_dependencies.energy_grid_lo)
        self.assertEqual(mock_energy_grid_hi_path, actual_dependencies.energy_grid_hi)
        self.assertEqual(mock_energy_grid_ultra_path, actual_dependencies.energy_grid_ultra)
        self.assertEqual(mock_tess_xyz_8_path, actual_dependencies.tess_xyz_8)
        self.assertEqual(mock_tess_ang_16_path, actual_dependencies.tess_ang16)
        self.assertEqual(fake_pipeline_settings_path, actual_dependencies.pipeline_settings_file)
        self.assertEqual(365, len(actual_dependencies.elongation))
        first_dict_value = actual_dependencies.elongation['2010001']
        last_dict_value = actual_dependencies.elongation['2010365']
        self.assertEqual(mock_lo_elongation_path, actual_dependencies.elongation_file)
        self.assertEqual(105, first_dict_value)
        self.assertEqual(105, last_dict_value)

        self.assertEqual(mock_downloaded_repoint_file, actual_dependencies.repointing_file)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.furnish_spice_metakernel')
    def test_furnish_spice_dependencies(self, mock_furnish_spice_metakernel):
        expected_kernel_types = [
            SpiceKernelTypes.ScienceFrames, SpiceKernelTypes.EphemerisReconstructed, SpiceKernelTypes.AttitudeHistory,
            SpiceKernelTypes.PointingAttitude, SpiceKernelTypes.PlanetaryEphemeris, SpiceKernelTypes.Leapseconds,
            SpiceKernelTypes.SpacecraftClock
        ]

        dependencies = self._create_l3e_dependencies()

        mock_furnish_spice_metakernel.return_value = FurnishMetakernelOutput(
            metakernel_path=Path("irrelevant.txt"),
            spice_kernel_paths=[Path("some_kernel_with_imap_data"), Path("some_kernel_with_planet_data")]
        )

        dependencies.furnish_spice_dependencies(sentinel.start_date, sentinel.end_date)

        mock_furnish_spice_metakernel.assert_called_once_with(
            start_date=sentinel.start_date,
            end_date=sentinel.end_date,
            kernel_types=expected_kernel_types
        )

        hi_parents = dependencies.get_hi_parents()
        self.assertIn("some_kernel_with_imap_data", hi_parents)
        self.assertIn("some_kernel_with_planet_data", hi_parents)

        lo_parents = dependencies.get_lo_parents()
        self.assertIn("some_kernel_with_imap_data", lo_parents)
        self.assertIn("some_kernel_with_planet_data", lo_parents)

        ul_parents = dependencies.get_ul_parents()
        self.assertIn("some_kernel_with_imap_data", ul_parents)
        self.assertIn("some_kernel_with_planet_data", ul_parents)


    def test_get_hi_parents(self):
        dependencies = self._create_l3e_dependencies()

        expected_parent_file_names = [
            "energy_grid_hi.dat",
            "lya_series.dat",
            "solar_uv_anisotropy.dat",
            "speed_3d_sw.dat",
            "density_3d_sw.dat",
            "phion_hydrogen.dat",
            "sw_eqtr_electrons.dat",
            "ionization_files.dat",
            "pipeline_settings.json",
            "repointing_file.csv",
        ]

        self.assertEqual(expected_parent_file_names, dependencies.get_hi_parents())

    def test_get_lo_parents(self):
        dependencies = self._create_l3e_dependencies()

        expected_parent_file_names = [
            "energy_grid_lo.dat",
            "tess_xyz_8.dat",
            "lya_series.dat",
            "solar_uv_anisotropy.dat",
            "speed_3d_sw.dat",
            "density_3d_sw.dat",
            "phion_hydrogen.dat",
            "sw_eqtr_electrons.dat",
            "ionization_files.dat",
            "pipeline_settings.json",
            "repointing_file.csv",
            "elongation.csv"
        ]

        self.assertEqual(expected_parent_file_names, dependencies.get_lo_parents())

    def test_get_ul_parents(self):
        dependencies = self._create_l3e_dependencies()

        expected_parent_file_names = [
            "energy_grid_ultra.dat",
            "tess_ang16.dat",
            "lya_series.dat",
            "solar_uv_anisotropy.dat",
            "speed_3d_sw.dat",
            "density_3d_sw.dat",
            "phion_hydrogen.dat",
            "sw_eqtr_electrons.dat",
            "ionization_files.dat",
            "pipeline_settings.json",
            "repointing_file.csv",
        ]

        self.assertEqual(expected_parent_file_names, dependencies.get_ul_parents())

    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.shutil')
    def test_rename_dependencies(self, mock_shutil):
        glows_l3e_dependencies = GlowsL3EDependencies(
            Path('2025/05/03/imap_glows_energy_grid_lo'),
            None,
            None,
            Path('2025/05/03/imap_glows_tess_xyz_8'),
            None,
            Path('2025/05/03/imap_glows_lya_series'),
            Path('2025/05/03/imap_glows_solar_uv_anisotropy'),
            Path('2025/05/03/imap_glows_speed_3d_sw'),
            Path('2025/05/03/imap_glows_density_3d_sw'),
            Path('2025/05/03/imap_glows_phion_hydrogen'),
            Path('2025/05/03/imap_glows_sw_eqtr_electrons'),
            Path('2025/05/03/imap_glows_ionization_files'),
            {
                "executable_dependency_paths": {
                    "energy-grid-lo": "EnGridLo.dat",
                    "energy-grid-hi": "EnGridHi.dat",
                    "energy-grid-ultra": "EnGridUltra.dat",
                    "tess-xyz-8": "tessXYZ8.dat",
                    "tess-ang-16": "tessAng16.dat",
                    "lya-series": "lyaSeriesV4_2021b.dat",
                    "solar-uv-anisotropy": "solar_uv_anisotropy_NP.1.0_SP.1.0.dat",
                    "speed-3d": "speed3D.v01.Legendre.2021b.dat",
                    "density-3d": "density3D.v01.Legendre.2021b.dat",
                    "phion-hydrogen": "phion_Hydrogen_T12F107_2021b.dat",
                    "ionization-files": "ionization.files.dat",
                    "sw-eqtr-electrons": "swEqtrElectrons5_2021b.dat",
                }
            },
            Path("path/to/some/pipeline_settings_file.csv"),
            {},
            Path("path/to/some/elongation_file.csv"),
            Path("repoint.csv")
        )

        expected_energy_grid_lo = 'EnGridLo.dat'
        expected_tess_xyz_8 = 'tessXYZ8.dat'
        expected_lya_series = 'lyaSeriesV4_2021b.dat'
        expected_solar_uv_anisotropy = 'solar_uv_anisotropy_NP.1.0_SP.1.0.dat'
        expected_speed_3d_sw = 'speed3D.v01.Legendre.2021b.dat'
        expected_density_3d_sw = 'density3D.v01.Legendre.2021b.dat'
        expected_phion_hydrogen = 'phion_Hydrogen_T12F107_2021b.dat'
        expected_sw_eqtr_electrons = 'swEqtrElectrons5_2021b.dat'
        expected_ionization_files = 'ionization.files.dat'

        glows_l3e_dependencies.rename_dependencies()

        mock_shutil.move.assert_has_calls([
            call(glows_l3e_dependencies.energy_grid_lo, expected_energy_grid_lo),
            call(glows_l3e_dependencies.tess_xyz_8, expected_tess_xyz_8),
            call(glows_l3e_dependencies.lya_series, expected_lya_series),
            call(glows_l3e_dependencies.solar_uv_anisotropy, expected_solar_uv_anisotropy),
            call(glows_l3e_dependencies.speed_3d_sw, expected_speed_3d_sw),
            call(glows_l3e_dependencies.density_3d_sw, expected_density_3d_sw),
            call(glows_l3e_dependencies.phion_hydrogen, expected_phion_hydrogen),
            call(glows_l3e_dependencies.sw_eqtr_electrons, expected_sw_eqtr_electrons),
            call(glows_l3e_dependencies.ionization_files, expected_ionization_files),
        ])

        mock_shutil.move.reset_mock()

        expected_energy_grid_hi = 'EnGridHi.dat'

        glows_l3e_dependencies.energy_grid_lo = None
        glows_l3e_dependencies.energy_grid_hi = Path('2025/05/03/imap_glows_energy_grid_hi')
        glows_l3e_dependencies.tess_xyz_8 = None

        glows_l3e_dependencies.rename_dependencies()

        mock_shutil.move.assert_has_calls([
            call(glows_l3e_dependencies.energy_grid_hi, expected_energy_grid_hi),
            call(glows_l3e_dependencies.lya_series, expected_lya_series),
            call(glows_l3e_dependencies.solar_uv_anisotropy, expected_solar_uv_anisotropy),
            call(glows_l3e_dependencies.speed_3d_sw, expected_speed_3d_sw),
            call(glows_l3e_dependencies.density_3d_sw, expected_density_3d_sw),
            call(glows_l3e_dependencies.phion_hydrogen, expected_phion_hydrogen),
            call(glows_l3e_dependencies.sw_eqtr_electrons, expected_sw_eqtr_electrons),
            call(glows_l3e_dependencies.ionization_files, expected_ionization_files),
        ])

        mock_shutil.move.reset_mock()

        glows_l3e_dependencies.energy_grid_hi = None
        glows_l3e_dependencies.energy_grid_ultra = Path('2025/05/03/imap_glows_energy_grid_ultra')
        glows_l3e_dependencies.tess_ang16 = Path('2025/05/03/imap_glows_tess_ang16')

        expected_energy_grid_ultra = 'EnGridUltra.dat'
        expected_tess_ang16 = 'tessAng16.dat'

        glows_l3e_dependencies.rename_dependencies()

        mock_shutil.move.assert_has_calls([
            call(glows_l3e_dependencies.energy_grid_ultra, expected_energy_grid_ultra),
            call(glows_l3e_dependencies.tess_ang16, expected_tess_ang16),
            call(glows_l3e_dependencies.lya_series, expected_lya_series),
            call(glows_l3e_dependencies.solar_uv_anisotropy, expected_solar_uv_anisotropy),
            call(glows_l3e_dependencies.speed_3d_sw, expected_speed_3d_sw),
            call(glows_l3e_dependencies.density_3d_sw, expected_density_3d_sw),
            call(glows_l3e_dependencies.phion_hydrogen, expected_phion_hydrogen),
            call(glows_l3e_dependencies.sw_eqtr_electrons, expected_sw_eqtr_electrons),
            call(glows_l3e_dependencies.ionization_files, expected_ionization_files),
        ])

    def _create_l3e_dependencies(self) -> GlowsL3EDependencies:
        return GlowsL3EDependencies(
            energy_grid_lo=Path("some/folder/energy_grid_lo.dat"),
            energy_grid_hi=Path("some/folder/energy_grid_hi.dat"),
            energy_grid_ultra=Path("some/folder/energy_grid_ultra.dat"),
            tess_xyz_8=Path("some/folder/tess_xyz_8.dat"),
            tess_ang16=Path("some/folder/tess_ang16.dat"),
            lya_series=Path("some/folder/lya_series.dat"),
            solar_uv_anisotropy=Path("some/folder/solar_uv_anisotropy.dat"),
            speed_3d_sw=Path("some/folder/speed_3d_sw.dat"),
            density_3d_sw=Path("some/folder/density_3d_sw.dat"),
            phion_hydrogen=Path("some/folder/phion_hydrogen.dat"),
            sw_eqtr_electrons=Path("some/folder/sw_eqtr_electrons.dat"),
            ionization_files=Path("some/folder/ionization_files.dat"),
            pipeline_settings_file=Path("some/folder/pipeline_settings.json"),
            pipeline_settings={},
            elongation_file=Path("some/folder/elongation.csv"),
            elongation={},
            repointing_file=Path("some/folder/repointing_file.csv"),
        )
