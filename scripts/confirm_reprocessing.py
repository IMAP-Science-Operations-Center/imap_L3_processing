import argparse
from datetime import datetime
from pathlib import Path

import imap_data_access


def report_oldest_file(**query_args):
    records = imap_data_access.query(**query_args, version="latest")
    oldest = min(records, key=lambda x:x["ingestion_date"])
    print(f"Oldest file:\n{oldest["ingestion_date"]}     {Path(oldest["file_path"]).name}")

def get_unprocessed_files(reference_time_str, **query_args):
    reference_time = datetime.fromisoformat(reference_time_str)
    records = imap_data_access.query(**query_args, version="latest")
    return [
        (r["start_date"], r["ingestion_date"])
        for r in records
        if datetime.fromisoformat(r["ingestion_date"]) < reference_time
    ]

def analyze(reference_time_str, **query_args):
    files = get_unprocessed_files(reference_time_str, **query_args)
    if len(files) == 0:
        print(f"All files have been reprocessed since {reference_time_str}")
    else:
        for date, processed_at in files:
            print(f"File {date} was last processed at {processed_at}")

parser = argparse.ArgumentParser()
parser.add_argument(
    "reprocessing_time",
    help="the time reprocessing was started, formatted as 20260915 or 20260915T17:01",
    nargs="?",
    default=None,
)
parser.add_argument("--instrument")
parser.add_argument("--data-level")
parser.add_argument("--descriptor")
args = parser.parse_args()
query_args = dict(instrument=args.instrument,
            descriptor=args.descriptor,
            data_level=args.data_level)
if args.reprocessing_time is not None:
    analyze(args.reprocessing_time,
            **query_args)
else:
    report_oldest_file(**query_args)