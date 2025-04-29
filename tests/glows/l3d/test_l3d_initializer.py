from unittest.mock import patch, Mock, call

from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies


@patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.query')
@patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.query')
def test_setup_and_run_toolkit(mock_query: Mock):
    l3d_dependencies: GlowsL3DDependencies = GlowsL3DDependencies(None, None, None)

    mock_query.side_effect = [
        [
            'imap_glows_l3b_ion-rate-profile_20100422.cdf',
            'imap_glows_l3b_ion-rate-profile_20100519.cdf'
            'imap_glows_l3b_ion-rate-profile_20100616.cdf'
        ],
        [
            'imap_glows_l3b_sw-profile_20100422.cdf',
            'imap_glows_l3b_sw-profile_20100519.cdf'
            'imap_glows_l3b_sw-profile_20100616.cdf'
        ],
    ]

    mock_query.assert_called_with([
        call(source="glows", descriptor="ion-rate-profile"),
        call(source="glows", descriptor="sw-profile"),
    ])
