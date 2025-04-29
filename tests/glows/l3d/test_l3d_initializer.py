from unittest.mock import patch, Mock, call

from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.l3d_initializer import setup_and_run_toolkit
from tests.test_helpers import get_test_data_path


@patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.get_l3a_parent_files_from_l3b')
@patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.query')
@patch('imap_l3_processing.glows.l3d.glows_l3d_initializer.download')
def test_setup_and_run_toolkit(mock_download: Mock, mock_query: Mock, mock_get_l3a_files):
    l3d_dependencies: GlowsL3DDependencies = GlowsL3DDependencies(
        get_test_data_path("glows/imap_glows_l3d_solar-param-hist_20100326_v000.cdf"), {}, {})

    mock_query.side_effect = [
        [
            'imap_glows_l3b_ion-rate-profile_20100422.cdf',
            'imap_glows_l3b_ion-rate-profile_20100519.cdf',
            'imap_glows_l3b_ion-rate-profile_20100616.cdf'
        ],
        [
            'imap_glows_l3b_sw-profile_20100422.cdf',
            'imap_glows_l3b_sw-profile_20100519.cdf',
            'imap_glows_l3b_sw-profile_20100616.cdf'
        ],
    ]

    mock_download.side_effect = [
        get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100422_v011.cdf'),
        get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100519_v011.cdf'),
        get_test_data_path('glows/imap_glows_l3c_sw-profile_20100422_v011.cdf'),
        get_test_data_path('glows/imap_glows_l3c_sw-profile_20100519_v011.cdf'),
    ]

    mock_get_l3a_files.side_effect = [
        [
            'imap_glows_l3a_hist_20100511-repoint00131_v011.cdf',
            'imap_glows_l3a_hist_20100512-repoint00132_v011.cdf'
        ],
        [
            'imap_glows_l3a_hist_20100518-repoint00138_v011.cdf',
            'imap_glows_l3a_hist_20100520-repoint00139_v011.cdf',
        ]
    ]

    setup_and_run_toolkit(l3d_dependencies)

    mock_get_l3a_files.assert_has_calls([
        call('imap_glows_l3b_ion-rate-profile_20100422.cdf'),
        call('imap_glows_l3b_ion-rate-profile_20100519.cdf')
    ])

    mock_query.assert_called_with([
        call(source="glows", descriptor="ion-rate-profile"),
        call(source="glows", descriptor="sw-profile"),
    ])

    mock_download.assert_called_with([
        call('imap_glows_l3b_ion-rate-profile_20100422.cdf'),
        call('imap_glows_l3b_ion-rate-profile_20100519.cdf'),
        call('imap_glows_l3b_sw-profile_20100422.cdf'),
        call('imap_glows_l3b_sw-profile_20100519.cdf'),
        call('imap_glows_l3a_hist_20100511-repoint00131_v011.cdf'),
        call('imap_glows_l3a_hist_20100512-repoint00132_v011.cdf'),
        call('imap_glows_l3a_hist_20100518-repoint00138_v011.cdf'),
        call('imap_glows_l3a_hist_20100520-repoint00139_v011.cdf'),
    ])
