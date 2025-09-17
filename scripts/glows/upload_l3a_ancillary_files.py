from pathlib import Path

from imap_data_access import upload

glows_file_dir = Path(__file__).parent.parent.parent / 'instrument_team_data' / 'glows'

upload(glows_file_dir / 'imap_glows_calibration-data_20250101_v002.dat')
# upload(glows_file_dir / 'imap_glows_map-of-extra-helio-bckgrd_20250101_v001.dat')
# upload(glows_file_dir / 'imap_glows_pipeline-settings_20250101_v001.json')
# upload(glows_file_dir / 'imap_glows_time-dep-bckgrd_20250101_v001.dat')
