import unittest
from unittest.mock import patch, call, Mock

from imap_l3_processing.glows.l3d.glows_l3d_initializer import GlowsL3DInitializer
from tests.test_helpers import create_glows_mock_query_results


class TestGlowsL3DInitializer(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer(self, mock_fetch_l3d_dependencies, mock_query, mock_read_cdf_parents):
        l3bs = [
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v001.cdf',
            'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
        ]
        l3cs = [
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200201-cr00002_v001.cdf',
            'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
        ]

        external_dependencies = Mock()

        mock_query.side_effect = [
            create_glows_mock_query_results([
                'imap_glows_l3d_solar-hist_19470301-cr00001_v000.cdf',
                'imap_glows_l3d_solar-hist_19470301-cr00001_v001.cdf',
                'imap_glows_l3d_solar-hist_19470301-cr00001_v002.cdf',
                'imap_glows_l3d_solar-hist_19470301-cr00002_v000.cdf',
                'imap_glows_l3d_solar-hist_19470301-cr00002_v001.cdf'
            ]),
            create_glows_mock_query_results(['imap_glows_plasma-speed-2010a_19470301_v001.dat']),
            create_glows_mock_query_results(['imap_glows_proton-density-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_uv-anisotropy-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_photoion-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_lya-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_electron-density-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_pipeline-settings-l3bcde_19470301_v000.json']),
        ]

        mock_read_cdf_parents.return_value = {
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200201-cr00002_v000.cdf',
            'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
            'imap_glows_plasma-speed-2010a_19470301_v001.dat',
            'imap_glows_proton-density-2010a_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2010a_19470301_v000.dat',
            'imap_glows_photoion-2010a_19470301_v000.dat',
            'imap_glows_lya-2010a_19470301_v000.dat',
            'imap_glows_electron-density-2010a_19470301_v000.dat',
            'imap_glows_pipeline-settings-l3bcde_19470301_v000.json',
            'lyman-alpha-composite.nc'
        }

        actual_version, actual_l3d_deps, actual_old_l3d = GlowsL3DInitializer.should_process_l3d(external_dependencies, l3bs, l3cs)

        most_recent_l3d = 'imap_glows_l3d_solar-hist_19470301-cr00002_v001.cdf'
        mock_read_cdf_parents.assert_called_once_with(most_recent_l3d)

        mock_query.assert_has_calls([
            call(instrument='glows', data_level="l3d", descriptor="solar-hist"),
            call(table='ancillary', instrument='glows', descriptor='plasma-speed-2010a', version='latest'),
            call(table='ancillary', instrument='glows', descriptor='proton-density-2010a', version='latest'),
            call(table='ancillary', instrument='glows', descriptor='uv-anisotropy-2010a', version='latest'),
            call(table='ancillary', instrument='glows', descriptor='photoion-2010a', version='latest'),
            call(table='ancillary', instrument='glows', descriptor='lya-2010a', version='latest'),
            call(table='ancillary', instrument='glows', descriptor='electron-density-2010a', version='latest'),
            call(table='ancillary', instrument='glows', descriptor='pipeline-settings-l3bcde', version='latest'),
        ])

        [fetch_dependencies_call] = mock_fetch_l3d_dependencies.call_args_list

        [actual_l3d_inputs, actual_external_deps] = fetch_dependencies_call.args

        pipeline_l3d_input_paths = actual_l3d_inputs.get_file_paths(source="glows")
        pipeline_l3d_input_filenames = [p.name for p in pipeline_l3d_input_paths]
        self.assertEqual([
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3b_ion-rate-profile_20200201-cr00002_v001.cdf',
            'imap_glows_l3b_ion-rate-profile_20200301-cr00003_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200201-cr00002_v001.cdf',
            'imap_glows_l3c_sw-profile_20200301-cr00003_v000.cdf',
            'imap_glows_plasma-speed-2010a_19470301_v001.dat',
            'imap_glows_proton-density-2010a_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2010a_19470301_v000.dat',
            'imap_glows_photoion-2010a_19470301_v000.dat',
            'imap_glows_lya-2010a_19470301_v000.dat',
            'imap_glows_electron-density-2010a_19470301_v000.dat',
            'imap_glows_pipeline-settings-l3bcde_19470301_v000.json'
        ], pipeline_l3d_input_filenames)

        self.assertEqual(external_dependencies, actual_external_deps)

        self.assertEqual(2, actual_version)
        self.assertEqual(mock_fetch_l3d_dependencies.return_value, actual_l3d_deps)
        self.assertEqual(most_recent_l3d, actual_old_l3d)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_returns_no_old_cdf_if_none_found(self, mock_fetch_l3d_deps, mock_query, mock_read_cdf_parents):
        l3bs = ['imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf']
        l3cs = ['imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf']

        external_deps = Mock()

        mock_query.side_effect = [
            [],
            create_glows_mock_query_results(['imap_glows_plasma-speed-2010a_19470301_v001.dat']),
            create_glows_mock_query_results(['imap_glows_proton-density-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_uv-anisotropy-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_photoion-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_lya-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_electron-density-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_pipeline-settings-l3bcde_19470301_v000.json']),
        ]

        _, __, old_l3d = GlowsL3DInitializer.should_process_l3d(external_deps, l3bs, l3cs)
        mock_read_cdf_parents.assert_not_called()
        mock_fetch_l3d_deps.assert_called_once()
        self.assertIsNone(old_l3d)

    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.read_cdf_parents')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.GlowsL3DDependencies.fetch_dependencies')
    def test_l3d_initializer_should_not_process(self, mock_fetch_l3d_deps, mock_query, mock_read_cdf_parents):
        l3bs = ['imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf']
        l3cs = ['imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf']

        external_dependencies = Mock()

        mock_query.side_effect = [
            create_glows_mock_query_results(['imap_glows_l3d_solar-hist_19470301-cr00002_v001.cdf']),
            create_glows_mock_query_results(['imap_glows_plasma-speed-2010a_19470301_v001.dat']),
            create_glows_mock_query_results(['imap_glows_proton-density-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_uv-anisotropy-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_photoion-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_lya-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_electron-density-2010a_19470301_v000.dat']),
            create_glows_mock_query_results(['imap_glows_pipeline-settings-l3bcde_19470301_v000.json']),
        ]

        mock_read_cdf_parents.return_value = {
            'imap_glows_l3b_ion-rate-profile_20200101-cr00001_v000.cdf',
            'imap_glows_l3c_sw-profile_20200101-cr00001_v000.cdf',
            'imap_glows_plasma-speed-2010a_19470301_v001.dat',
            'imap_glows_proton-density-2010a_19470301_v000.dat',
            'imap_glows_uv-anisotropy-2010a_19470301_v000.dat',
            'imap_glows_photoion-2010a_19470301_v000.dat',
            'imap_glows_lya-2010a_19470301_v000.dat',
            'imap_glows_electron-density-2010a_19470301_v000.dat',
            'imap_glows_pipeline-settings-l3bcde_19470301_v000.json',
            'lyman-alpha-composite.nc'
        }

        actual_l3d_deps = GlowsL3DInitializer.should_process_l3d(external_dependencies, l3bs, l3cs)
        mock_fetch_l3d_deps.assert_not_called()
        self.assertIsNone(actual_l3d_deps)
