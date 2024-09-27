import argparse
import json
from datetime import datetime

from imap_processing.models import UpstreamDataDependency, InputMetadata
from imap_processing.swapi.swapi_processor import SwapiProcessor


def imap_l3_processor():
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

    if args.instrument == 'swapi' and (args.data_level == 'l3a' or args.data_level == 'l3b'):
        dependencies = [UpstreamDataDependency(d['instrument'], d['data_level'],
                                               None, None,
                                               d['version'], d['descriptor']) for d in dependencies_list]
        input_dependency = InputMetadata(args.instrument,
                                         args.data_level,
                                         datetime.strptime(args.start_date, '%Y%m%d'),
                                         datetime.strptime(args.end_date,
                                                           '%Y%m%d') if args.end_date is not None else None,
                                         args.version)
        processor = SwapiProcessor(dependencies, input_dependency)
        processor.process()
    else:
        raise NotImplementedError(
            f'Level {args.data_level} data processing has not yet been implemented for {args.instrument}')


if __name__ == '__main__':
    imap_l3_processor()
