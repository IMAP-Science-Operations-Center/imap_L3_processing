import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import scripts
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_from_dictionary
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import save_data
from tests.test_helpers import get_test_data_path

input_file_directory = "/Users/harrison/Documents/l3a"

files_to_convert = os.listdir(input_file_directory)
for file in files_to_convert:
    with open(Path(input_file_directory) / file, "r") as l3a_json:
        data = json.load(l3a_json)
    date = file.split('_')[3][:8]
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    start_date = datetime(year, month, day)
    input_metadata = InputMetadata(
        instrument='glows',
        data_level='l3a',
        start_date=start_date,
        end_date=start_date + timedelta(days=1),
        version='v001',
        descriptor='hist'
    )

    files = {
        "settings":
            Path(scripts.__file__).parent.parent /
            "instrument_team_data/glows/imap_glows_pipeline-settings_20100101_v001.json"
    }
    data_with_spin_angle = GlowsProcessor.add_spin_angle_delta(data, files)

    glows_l3a_lightcurve = create_glows_l3a_from_dictionary(data_with_spin_angle,
                                                            input_metadata)

    cdf_path = save_data(glows_l3a_lightcurve, delete_if_present=True,
                         folder_path=get_test_data_path("glows/l3a_products"))

    print("cdf written to", cdf_path)
