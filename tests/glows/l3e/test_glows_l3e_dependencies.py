import unittest
from pathlib import Path
from unittest.mock import patch, Mock, call

from imap_data_access import RepointInput

from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from tests.test_helpers import get_test_data_path


class TestGlowsL3EDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.download_dependency_from_path')
    def test_fetch_dependencies_handles_lo(self, mock_download_dependency_from_path):
        mock_processing_input_collection = Mock()

        mock_l3d = Path('imap_glows_l3d_solar-hist_20250501-cr02091_v001.cdf')
        mock_energy_grid_lo = Path('energy_grid_lo_path')
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
        mock_repoint_file = Path('repoint.csv')

        mock_processing_input_collection.get_file_paths.side_effect = [
            [mock_l3d], [mock_lya_series], [mock_solar_uv_anisotropy], [mock_speed_3d_sw], [mock_density_3d_sw],
            [mock_phion_hydrogen], [mock_sw_eqtr_electrons], [ionization_files], [mock_pipeline_settings],
            [mock_tess_xyz_8], [mock_energy_grid_lo], [mock_lo_elongation_file], [mock_repoint_file]
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
        mock_downloaded_repoint_file = Mock()

        mock_download_dependency_from_path.side_effect = [
            mock_lya_series_path, mock_solar_uv_anisotropy_path, mock_speed_3d_sw_path,
            mock_density_3d_sw_path, mock_phion_hydrogen_path, mock_sw_eqtr_electrons_path, mock_ionization_files_path,
            fake_pipeline_settings_path, mock_energy_grid_lo_path, mock_tess_xyz_8_path, mock_lo_elongation_path,
            mock_downloaded_repoint_file,
        ]

        actual_dependencies, cr_number = GlowsL3EDependencies.fetch_dependencies(
            mock_processing_input_collection, "survival-probability-lo")

        self.assertEqual(13, mock_processing_input_collection.get_file_paths.call_count)

        mock_processing_input_collection.get_file_paths.assert_has_calls([
            call(source="glows", descriptor="solar-hist"),
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
            call(data_type=RepointInput.data_type),
        ])

        mock_download_dependency_from_path.assert_has_calls([
            call('lya_series_sdc_path'), call('solar_uv_anisotropy_sdc_path'),
            call('speed_3d_sw_sdc_path'), call('density_3d_sw_sdc_path'), call('phion_hydrogen_sdc_path'),
            call('sw_eqtr_electrons_sdc_path'), call('ionization_files_path'),
            call(mock_pipeline_settings.name),
            call('tess_xyz_8_sdc_path'),
            call('energy_grid_lo_path'),
            call('lo_elongation_data.dat'),
            call('repoint.csv'),
        ])

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
        self.assertEqual(None, actual_dependencies.energy_grid_hi)
        self.assertEqual(None, actual_dependencies.energy_grid_ultra)
        self.assertEqual(mock_tess_xyz_8_path, actual_dependencies.tess_xyz_8)
        self.assertEqual(None, actual_dependencies.tess_ang16)
        self.assertEqual(365, len(actual_dependencies.elongation))
        first_dict_value = actual_dependencies.elongation['2010001']
        last_dict_value = actual_dependencies.elongation['2010365']
        self.assertEqual(105, first_dict_value)
        self.assertEqual(105, last_dict_value)

        self.assertEqual(cr_number, 2091)
        self.assertEqual(mock_downloaded_repoint_file, actual_dependencies.repointing_file)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.download_dependency_from_path')
    def test_fetch_dependencies_handles_hi(self, mock_download_dependency_from_path):
        test_cases = ["survival-probability-hi-45", "survival-probability-hi-90"]

        for descriptor in test_cases:
            with self.subTest(descriptor):
                mock_processing_input_collection = Mock()

                mock_l3d = Path('imap_glows_l3d_solar-hist_20250501-repoint00004_v001.cdf')
                mock_energy_grid_hi = Path('energy_grid_hi_sdc_path')
                mock_lya_series = Path('lya_series_sdc_path')
                mock_solar_uv_anisotropy = Path('solar_uv_anisotropy_sdc_path')
                mock_speed_3d_sw = Path('speed_3d_sw_sdc_path')
                mock_density_3d_sw = Path('density_3d_sw_sdc_path')
                mock_phion_hydrogen = Path('phion_hydrogen_sdc_path')
                mock_sw_eqtr_electrons = Path('sw_eqtr_electrons_sdc_path')
                ionization_files = Path('ionization_files_path')
                mock_pipeline_settings = Path('l3bcde_pipeline_settings.json')
                mock_repoint_file = Path('repoint.csv')

                mock_processing_input_collection.get_file_paths.side_effect = [
                    [mock_l3d], [mock_lya_series], [mock_solar_uv_anisotropy], [mock_speed_3d_sw], [mock_density_3d_sw],
                    [mock_phion_hydrogen], [mock_sw_eqtr_electrons], [ionization_files], [mock_pipeline_settings],
                    [mock_energy_grid_hi], [mock_repoint_file]
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
                mock_energy_grid_hi_path = Mock()
                mock_downloaded_repoint_file = Mock()

                mock_download_dependency_from_path.side_effect = [mock_lya_series_path, mock_solar_uv_anisotropy_path,
                                                                  mock_speed_3d_sw_path,
                                                                  mock_density_3d_sw_path, mock_phion_hydrogen_path,
                                                                  mock_sw_eqtr_electrons_path,
                                                                  mock_ionization_files_path,
                                                                  fake_pipeline_settings_path, mock_energy_grid_hi_path,
                                                                  mock_downloaded_repoint_file
                                                                  ]

                actual_dependencies, repointing = GlowsL3EDependencies.fetch_dependencies(
                    mock_processing_input_collection, descriptor)

                mock_processing_input_collection.get_file_paths.assert_has_calls([
                    call(source="glows", descriptor="solar-hist"),
                    call(source="glows", descriptor="lya"),
                    call(source="glows", descriptor="uv-anis"),
                    call(source="glows", descriptor="speed"),
                    call(source="glows", descriptor="p-dens"),
                    call(source="glows", descriptor="phion"),
                    call(source="glows", descriptor="e-dens"),
                    call(source="glows", descriptor="ionization-files"),
                    call(source="glows", descriptor="pipeline-settings-l3bcde"),
                    call(source="glows", descriptor="energy-grid-hi"),
                    call(data_type=RepointInput.data_type),
                ])
                self.assertEqual(11, mock_processing_input_collection.get_file_paths.call_count)

                mock_download_dependency_from_path.assert_has_calls([call('lya_series_sdc_path'),
                                                                     call('solar_uv_anisotropy_sdc_path'),
                                                                     call('speed_3d_sw_sdc_path'),
                                                                     call('density_3d_sw_sdc_path'),
                                                                     call('phion_hydrogen_sdc_path'),
                                                                     call('sw_eqtr_electrons_sdc_path'),
                                                                     call('ionization_files_path'),
                                                                     call(mock_pipeline_settings.name),
                                                                     call('energy_grid_hi_sdc_path'),
                                                                     call('repoint.csv'),
                                                                     ])

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

                self.assertEqual(actual_dependencies.lya_series, mock_lya_series_path)
                self.assertEqual(actual_dependencies.solar_uv_anisotropy, mock_solar_uv_anisotropy_path)
                self.assertEqual(actual_dependencies.speed_3d_sw, mock_speed_3d_sw_path)
                self.assertEqual(actual_dependencies.density_3d_sw, mock_density_3d_sw_path)
                self.assertEqual(actual_dependencies.phion_hydrogen, mock_phion_hydrogen_path)
                self.assertEqual(actual_dependencies.sw_eqtr_electrons, mock_sw_eqtr_electrons_path)
                self.assertEqual(actual_dependencies.pipeline_settings['executable_dependency_paths'],
                                 expected_pipeline_settings['executable_dependency_paths'])
                self.assertEqual(actual_dependencies.ionization_files, mock_ionization_files_path)
                self.assertEqual(actual_dependencies.energy_grid_lo, None)
                self.assertEqual(actual_dependencies.energy_grid_hi, mock_energy_grid_hi_path)
                self.assertEqual(actual_dependencies.energy_grid_ultra, None)
                self.assertEqual(actual_dependencies.tess_xyz_8, None)
                self.assertEqual(actual_dependencies.tess_ang16, None)
                self.assertEqual(repointing, 4)
                self.assertEqual(mock_downloaded_repoint_file, actual_dependencies.repointing_file)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.download_dependency_from_path')
    def test_fetch_dependencies_handles_ultra(self, mock_download_dependency_from_path):
        mock_processing_input_collection = Mock()

        mock_l3d = Path('imap_glows_l3d_solar-hist_20250501-repoint00004_v001.cdf')
        mock_energy_grid_ul = Path('energy_grid_ul_path')
        mock_tess_ang_16 = Path('tess_ang_16_path')
        mock_lya_series = Path('lya_series_sdc_path')
        mock_solar_uv_anisotropy = Path('solar_uv_anisotropy_sdc_path')
        mock_speed_3d_sw = Path('speed_3d_sw_sdc_path')
        mock_density_3d_sw = Path('density_3d_sw_sdc_path')
        mock_phion_hydrogen = Path('phion_hydrogen_sdc_path')
        mock_sw_eqtr_electrons = Path('sw_eqtr_electrons_sdc_path')
        ionization_files = Path('ionization_files_path')
        mock_pipeline_settings = Path('l3e_pipeline_settings.json')
        mock_repoint_file = Path('repoint.csv')

        mock_processing_input_collection.get_file_paths.side_effect = [
            [mock_l3d], [mock_lya_series], [mock_solar_uv_anisotropy], [mock_speed_3d_sw],
            [mock_density_3d_sw],
            [mock_phion_hydrogen], [mock_sw_eqtr_electrons], [ionization_files], [mock_pipeline_settings],
            [mock_energy_grid_ul],
            [mock_tess_ang_16], [mock_repoint_file]
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
        mock_energy_grid_ul_path = Mock()
        mock_tess_ang_16_path = Mock()
        mock_downloaded_repoint_path = Mock()

        mock_download_dependency_from_path.side_effect = [mock_lya_series_path, mock_solar_uv_anisotropy_path,
                                                          mock_speed_3d_sw_path,
                                                          mock_density_3d_sw_path, mock_phion_hydrogen_path,
                                                          mock_sw_eqtr_electrons_path,
                                                          mock_ionization_files_path, fake_pipeline_settings_path,
                                                          mock_energy_grid_ul_path, mock_tess_ang_16_path,
                                                          mock_downloaded_repoint_path
                                                          ]

        actual_dependencies, repointing = GlowsL3EDependencies.fetch_dependencies(
            mock_processing_input_collection, "survival-probability-ul")

        mock_processing_input_collection.get_file_paths.assert_has_calls([
            call(source="glows", descriptor="solar-hist"),
            call(source="glows", descriptor="lya"),
            call(source="glows", descriptor="uv-anis"),
            call(source="glows", descriptor="speed"),
            call(source="glows", descriptor="p-dens"),
            call(source="glows", descriptor="phion"),
            call(source="glows", descriptor="e-dens"),
            call(source="glows", descriptor="ionization-files"),
            call(source="glows", descriptor="pipeline-settings-l3bcde"),
            call(source="glows", descriptor="energy-grid-ultra"),
            call(source="glows", descriptor="tess-ang-16"),
            call(data_type=RepointInput.data_type),
        ])
        self.assertEqual(12, mock_processing_input_collection.get_file_paths.call_count)

        mock_download_dependency_from_path.assert_has_calls(
            [call('lya_series_sdc_path'), call('solar_uv_anisotropy_sdc_path'),
             call('speed_3d_sw_sdc_path'), call('density_3d_sw_sdc_path'), call('phion_hydrogen_sdc_path'),
             call('sw_eqtr_electrons_sdc_path'), call('ionization_files_path'),
             call(mock_pipeline_settings.name),
             call('energy_grid_ul_path'),
             call('tess_ang_16_path'),
             call('repoint.csv'),
             ])

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

        self.assertEqual(actual_dependencies.lya_series, mock_lya_series_path)
        self.assertEqual(actual_dependencies.solar_uv_anisotropy, mock_solar_uv_anisotropy_path)
        self.assertEqual(actual_dependencies.speed_3d_sw, mock_speed_3d_sw_path)
        self.assertEqual(actual_dependencies.density_3d_sw, mock_density_3d_sw_path)
        self.assertEqual(actual_dependencies.phion_hydrogen, mock_phion_hydrogen_path)
        self.assertEqual(actual_dependencies.sw_eqtr_electrons, mock_sw_eqtr_electrons_path)
        self.assertEqual(actual_dependencies.pipeline_settings['executable_dependency_paths'],
                         expected_pipeline_settings['executable_dependency_paths'])
        self.assertEqual(actual_dependencies.ionization_files, mock_ionization_files_path)
        self.assertEqual(actual_dependencies.energy_grid_lo, None)
        self.assertEqual(actual_dependencies.energy_grid_hi, None)
        self.assertEqual(actual_dependencies.energy_grid_ultra, mock_energy_grid_ul_path)
        self.assertEqual(actual_dependencies.tess_xyz_8, None)
        self.assertEqual(actual_dependencies.tess_ang16, mock_tess_ang_16_path)
        self.assertEqual(repointing, 4)
        self.assertEqual(mock_downloaded_repoint_path, actual_dependencies.repointing_file)

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
            {},
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
