from datetime import datetime

from imap_processing.models import UpstreamDataDependency, InputMetadata
from imap_processing.swapi.l3a.processor import SwapiL3AProcessor
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--instrument")
parser.add_argument("--level")
parser.add_argument("--start_date")
parser.add_argument("--end_date")
parser.add_argument("--version")
parser.add_argument("--dependency")

args = parser.parse_args()
dependencies_list = json.loads(args.dependency.replace("'", '"'))

if args.instrument == 'swapi':
    if args.level == 'l3a':
        dependencies = [UpstreamDataDependency(d['instrument'], d['data_level'], d['descriptor'],
                                               None, None,
                                               d['version']) for d in dependencies_list]
        input_dependency = InputMetadata(args.instrument,
                                         args.level,
                                         datetime.strptime(args.start_date, '%Y%m%d'),
                                         datetime.strptime(args.end_date, '%Y%m%d'),
                                         args.version)
        processor = SwapiL3AProcessor(dependencies, input_dependency)
        processor.process()
else:
    raise NotImplemented(f'Level {args.level} data processing has not yet been implemented for {args.instrument}')
