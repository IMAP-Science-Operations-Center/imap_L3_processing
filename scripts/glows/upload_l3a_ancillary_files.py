from imap_data_access import upload

from tests.test_helpers import get_test_instrument_team_data_path

glows_file_dir = get_test_instrument_team_data_path('glows')

upload(glows_file_dir / 'imap_glows_calibration-data_20100101_v002.dat')
upload(glows_file_dir / 'imap_glows_map-of-extra-helio-bckgrd_20100101_v002.dat')
upload(glows_file_dir / 'imap_glows_pipeline-settings_20100101_v001.json')
upload(glows_file_dir / 'imap_glows_time-dep-bckgrd_20100101_v001.dat')
