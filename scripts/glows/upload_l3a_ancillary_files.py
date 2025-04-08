from pathlib import Path

from imap_data_access import upload

glows_file_dir = Path(__file__).parent.parent.parent / 'instrument_team_data' / 'glows'

upload(glows_file_dir / 'imap_glows_l3a_calibration-data-text-not-cdf_20250707_v002.cdf')
upload(glows_file_dir / 'imap_glows_l3a_map-of-extra-helio-bckgrd-text-not-cdf_20250707_v001.cdf')
upload(glows_file_dir / 'imap_glows_l3a_pipeline-settings-json-not-cdf_20250707_v002.cdf')
upload(glows_file_dir / 'imap_glows_l3a_time-dep-bckgrd-text-not-cdf_20250707_v001.cdf')
