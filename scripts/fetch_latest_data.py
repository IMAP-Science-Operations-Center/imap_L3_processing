import argparse

import imap_data_access

parser = argparse.ArgumentParser()
parser.add_argument("--instrument", required=True)
parser.add_argument("--level", required=True)
parser.add_argument("--count", required=True)

args = parser.parse_args()

data = imap_data_access.query(instrument=args.instrument, data_level=args.level)
sorted_data = sorted(data, key=lambda r: r['ingestion_date'])
paths = [f['file_path'] for f in sorted_data[-int(args.count):]]
for path in paths:
    imap_data_access.download(path)
    print(path)
