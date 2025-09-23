import enum
import os
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

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


class Command(enum.Enum):
    Query = "query"
    Logs = "logs"

arg_parser = ArgumentParser()

subparsers = arg_parser.add_subparsers()

query_parser = subparsers.add_parser('query')
queryable_fields = ["status", "instrument", "data_level", "descriptor", "start_date", "version", "repointing",
                            "job_definition", "job_log_stream_id", "container_image", "container_command",
                            "started_at", "stopped_at"]
for field in queryable_fields:
    query_parser.add_argument(f"--{field}", type=str)


log_parser = subparsers.add_parser('logs')
log_parser.add_argument('log_id', type=str)

args = arg_parser.parse_args()
print(args)

imap_dev_url = os.getenv("IMAP_DATA_ACCESS_URL")
headers = {'x-api-key': os.getenv("IMAP_API_KEY")}

if not "log_id" in vars(args):
    query_params = { key: val for key, val in vars(args).items() if val is not None and key not in ['command']}
    response = requests.get(url=f"{imap_dev_url}/processing-jobs", headers=headers, params=query_params)
    processing_jobs = response.json()
    records = []
    for processing_job in processing_jobs:
        if processing_job["started_at"] is not None:
            processing_job["started_at"] = datetime.fromisoformat(processing_job["started_at"]).strftime("%Y-%m-%d %H:%M:%S")

        relevant_fields = [
            "status",
            "instrument",
            "data_level",
            "descriptor",
            "start_date",
            "version",
            "started_at",
            "job_log_stream_id"
        ]

        records.append({key: val for key, val in processing_job.items() if key in relevant_fields})

    pd.set_option('display.max_colwidth', None)
    print(pd.DataFrame.from_records(records).sort_values('started_at').tail(50))
else:
    logs = requests.get(url=f"{imap_dev_url}/processing-logs/{args.log_id}", headers=headers)
    print(logs.text)


