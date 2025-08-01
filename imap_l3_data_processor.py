import argparse
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import imap_data_access
import spiceypy
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.codice_hi_processor import CodiceHiProcessor
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hit.l3.hit_processor import HitProcessor
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from imap_l3_processing.swe.swe_processor import SweProcessor
from imap_l3_processing.ultra.l3.ultra_processor import UltraProcessor


def _parse_cli_arguments():
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

    return parser.parse_args()


def _convert_to_datetime(date):
    if date is None:
        return None
    else:
        return datetime.strptime(date, "%Y%m%d")


def imap_l3_processor():
    args = _parse_cli_arguments()

    # If the dependency argument was passed in as a json file, read it into a string
    if args.dependency.endswith(".json"):
        dependency_filepath = imap_data_access.download(args.dependency)
        with open(dependency_filepath) as f:
            args.dependency = f.read()

    processing_input_collection = ProcessingInputCollection()
    processing_input_collection.deserialize(args.dependency)

    _furnish_spice_kernels(processing_input_collection)
    input_dependency = InputMetadata(args.instrument,
                                     args.data_level,
                                     _convert_to_datetime(args.start_date),
                                     _convert_to_datetime(args.end_date or args.start_date),
                                     args.version, descriptor=args.descriptor)
    if args.instrument == 'swapi' and (args.data_level == 'l3a' or args.data_level == 'l3b'):
        processor = SwapiProcessor(processing_input_collection, input_dependency)
    elif args.instrument == 'glows':
        processor = GlowsProcessor(processing_input_collection, input_dependency)
    elif args.instrument == 'swe' and args.data_level == 'l3':
        processor = SweProcessor(processing_input_collection, input_dependency)
    elif args.instrument == 'hit':
        processor = HitProcessor(processing_input_collection, input_dependency)
    elif args.instrument == 'hi':
        processor = HiProcessor(processing_input_collection, input_dependency)
    elif args.instrument == 'ultra':
        processor = UltraProcessor(processing_input_collection, input_dependency)
    elif args.instrument == 'lo':
        processor = LoProcessor(processing_input_collection, input_dependency)
    elif args.instrument == 'codice':
        if args.descriptor.startswith("hi"):
            processor = CodiceHiProcessor(processing_input_collection, input_dependency)
        elif args.descriptor.startswith("lo"):
            processor = CodiceLoProcessor(processing_input_collection, input_dependency)
        else:
            raise NotImplementedError(f"Unknown descriptor '{args.descriptor}' for codice instrument")
    else:
        raise NotImplementedError(
            f'Level {args.data_level} data processing has not yet been implemented for {args.instrument}')

    paths = processor.process()
    if args.upload_to_sdc:
        for path in paths:
            imap_data_access.upload(path)


def _furnish_spice_kernels(processing_input_collection):
    spice_kernel_paths = processing_input_collection.get_file_paths(data_type='spice')
    for kernel in spice_kernel_paths:
        kernel_path = imap_data_access.download(kernel)
        spiceypy.furnsh(str(kernel_path))


if __name__ == '__main__':
    with TemporaryDirectory() as dir:
        args = _parse_cli_arguments()
        logger = logging.getLogger('application')
        logger.setLevel(logging.INFO)

        log_path = Path(
            dir) / f"imap_{args.instrument}_{args.data_level}_log-{datetime.now().strftime('%Y-%m-%d-%H%M%S-%f')}_{args.start_date}_v001.cdf"
        fh = logging.FileHandler(str(log_path))

        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        try:
            imap_l3_processor()
        except Exception as e:
            logger.info("Unhandled Exception:", exc_info=e)
            print("caught exception")
            traceback.print_exc()
            logging.shutdown()
            raise e
        finally:
            should_upload_log = False
            if should_upload_log and os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                imap_data_access.upload(log_path)

            logging.shutdown()
