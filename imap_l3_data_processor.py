from datetime import datetime

from imap_processing.models import UpstreamDataDependency, InputMetadata
from imap_processing.swapi.processor import SwapiProcessor
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--instrument")
parser.add_argument("--data-level")
parser.add_argument("--descriptor")
parser.add_argument("--start-date")
parser.add_argument("--end-date", required=False)
parser.add_argument("--version")
parser.add_argument("--dependency")
parser.add_argument(
    "--upload-to-sdc",
    action="store_true",
    required=False,
    help="Upload completed output files to the IMAP SDC.",
)

args = parser.parse_args()
dependencies_list = json.loads(args.dependency.replace("'", '"'))

if args.instrument == 'swapi':
    if args.data_level == 'l3a' or args.data_level == 'l3b':
        dependencies = [UpstreamDataDependency(d['instrument'], d['data_level'],
                                               None, None,
                                               d['version'], d['descriptor']) for d in dependencies_list]
        input_dependency = InputMetadata(args.instrument,
                                         args.data_level,
                                         datetime.strptime(args.start_date, '%Y%m%d'),
                                         None,
                                         args.version)
        processor = SwapiProcessor(dependencies, input_dependency)
        processor.process()
else:
    raise NotImplemented(f'Level {args.level} data processing has not yet been implemented for {args.instrument}')
