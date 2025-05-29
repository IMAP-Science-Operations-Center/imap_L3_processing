import json
import time
from datetime import datetime

import pytz
import requests
from imap_data_access import ProcessingInputCollection


def format_job(job) -> str:
    dependencies_passed = job["container_command"].split("--dependency ")[1].split(" --upload-to-sdc")[0]

    input_collection = ProcessingInputCollection()
    input_collection.deserialize(dependencies_passed)

    processing_input = "\n".join([f"\t\t{type(pi)}: {pi.filename_list}" for pi in input_collection.processing_input])

    return f"""
        descriptor: {job["descriptor"]}
        start_date: {job["start_date"]}
        version: {job['version']}
        status: {job["status"]}
        processing_input: 
{processing_input}
    """


INSTRUMENT = "codice"
DATA_LEVEL = "l3a"

AFTER_DATE = datetime.now(tz=pytz.UTC)

url = 'https://api.dev.imap-mission.com/batch-job'

while True:
    response = requests.get(url=url, params={'instrument': INSTRUMENT, 'data_level': DATA_LEVEL})
    logs = json.loads(json.dumps(response.json()))
    relevant_logs = [log for log in logs if
                     log['started_at'] is not None and
                     datetime.fromisoformat(log['started_at']).replace(tzinfo=pytz.UTC) > AFTER_DATE
                     ]

    print("\n".join([format_job(job) for job in relevant_logs]))

    time.sleep(15)
