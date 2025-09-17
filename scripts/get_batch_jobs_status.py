import enum
import json
from argparse import ArgumentParser

import requests

"""
    "status": "SUCCEEDED",
    "instrument": "glows",
    "data_level": "l3b",
    "descriptor": "ion-rate-profile",
    "start_date": "2025-09-17T00:00:00",
    "version": "v001",
    "repointing": null,
    "job_definition": "arn:aws:batch:us-west-2:449431850278:job-definition/ProcessingJob-glows-l3:15",
    "job_log_stream_id": "ProcessingJob-glows-l3/default/224ab38452544d0faf5299710c96c77f",
    "container_image": "449431850278.dkr.ecr.us-west-2.amazonaws.com/glows-l3-repo:latest",
    "container_command": "--instrument glows --data-level l3b --descriptor ion-rate-profile --start-date 20250917 --version v001 --dependency imap_glows_l3b_ion-rate-profile-ed9a3e71_20250917_v001.json --upload-to-sdc",
    "started_at": "2025-09-17T14:48:47.283000+00:00",
    "stopped_at": "2025-09-17T14:51:59.784000+00:00"
"""

class ResultView(enum.Enum):
    Count = "count"
    Filenames = "filenames"



available_fields = [
    "status",
    "instrument",
    "data_level",
    "descriptor",
    "start_date",
    "version",
    "repointing",
    "job_definition",
    "job_log_stream_id",
    "container_image",
    "container_command",
    "started_at",
    "stopped_at",
]

non_query_arguments = [
    'api_key',
    'view'
]

arg_parser = ArgumentParser()

arg_parser.add_argument('api_key', type=str)
for field in available_fields:
    arg_parser.add_argument(f"--{field}", type=str)

args = arg_parser.parse_args()

imap_dev_url = "https://api.dev.imap-mission.com/api-key"

url = f'{imap_dev_url}/processing-jobs'

query_params = { key: val for key, val in vars(args).items() if val is not None and key != 'api_key'}
headers = {'x-api-key': args.api_key}
response = requests.get(url=url, headers=headers, params=query_params)

processing_jobs = response.json()
for processing_job in processing_jobs:
    print(processing_job['container_command'], processing_job['started_at'], processing_job['stopped_at'])
    logs = requests.get(url=f"{imap_dev_url}/processing-logs/{processing_job['job_log_stream_id']}", headers=headers)
    print(logs.text)

# reverse_logs = sorted(logs, key=lambda x: x['started_at'] or "")
# filtered_logs = list(filter(lambda x: x["descriptor"] == DESCRIPTOR, logs))

