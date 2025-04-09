import argparse
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import imap_data_access
from imap_data_access import ScienceFilePath
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.hit.l3.hit_processor import HitProcessor
from imap_l3_processing.models import UpstreamDataDependency, InputMetadata
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from imap_l3_processing.swe.swe_processor import SweProcessor


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
    processing_input_collection = ProcessingInputCollection()
    processing_input_collection.deserialize(args.dependency)

    def convert_to_datetime(date):
        if date is None:
            return None
        else:
            return datetime.strptime(date, "%Y%m%d")

    dependencies = []
    for dependency in processing_input_collection.get_science_inputs():
        for file_path in dependency.imap_file_paths:
            file_path: ScienceFilePath
            dependencies.append(UpstreamDataDependency(
                file_path.instrument, file_path.data_level, convert_to_datetime(file_path.start_date),
                convert_to_datetime(file_path.start_date), file_path.version, file_path.descriptor
            ))

    input_dependency = InputMetadata(args.instrument,
                                     args.data_level,
                                     convert_to_datetime(args.start_date),
                                     convert_to_datetime(args.end_date or args.start_date),
                                     args.version, descriptor=args.descriptor)
    if args.instrument == 'swapi' and (args.data_level == 'l3a' or args.data_level == 'l3b'):
        processor = SwapiProcessor(dependencies, input_dependency)
        processor.process()
    elif args.instrument == 'glows':
        processor = GlowsProcessor(dependencies, input_dependency)
        processor.process()
    elif args.instrument == 'swe' and args.data_level == 'l3':
        processor = SweProcessor(dependencies, input_dependency)
        processor.process()
    elif args.instrument == 'hit':
        processor = HitProcessor(dependencies, input_dependency)
        processor.process()
    else:
        raise NotImplementedError(
            f'Level {args.data_level} data processing has not yet been implemented for {args.instrument}')


if __name__ == '__main__':
    with TemporaryDirectory() as dir:
        logger = logging.getLogger(__name__)
        log_path = Path(
            dir) / f"imap_swapi_l3_log-{datetime.now().strftime('%Y-%m-%d-%H%M%S-%f')}_20240606_v001.cdf"

        logging.basicConfig(filename=str(log_path), level=logging.INFO)
        logger.info('Started')

        try:
            imap_l3_processor()
        except Exception as e:
            logger.error("Unhandled Exception:", exc_info=e)
            print("caught exception")
            traceback.print_exc()

        logging.shutdown()
        imap_data_access.upload(log_path)
