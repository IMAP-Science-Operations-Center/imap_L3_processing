import argparse
import logging
import re
from datetime import datetime
from tempfile import TemporaryDirectory

import imap_data_access
import spiceypy
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.codice_hi_processor import CodiceHiProcessor
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.l3.hi_l3_initializer import HiL3Initializer, HI_SP_MAP_DESCRIPTORS
from imap_l3_processing.hit.l3.hit_processor import HitProcessor
from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from imap_l3_processing.swe.swe_processor import SweProcessor
from imap_l3_processing.ultra.l3.ultra_processor import UltraProcessor

logger = logging.getLogger(__name__)


def _parse_cli_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument")
    parser.add_argument("--data-level")
    parser.add_argument("--descriptor")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date", required=False)
    parser.add_argument("--repointing", required=False)
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

    repointing_number = None
    if args.repointing is not None:
        repointing_number_match = re.match(r"repoint(?P<repoint>\d{5})", args.repointing)
        if repointing_number_match is None:
            raise ValueError("Unexpected repointing number command line format!")
        repointing_number = int(repointing_number_match["repoint"])

    _furnish_spice_kernels(processing_input_collection)
    input_dependency = InputMetadata(args.instrument,
                                     args.data_level,
                                     _convert_to_datetime(args.start_date),
                                     _convert_to_datetime(args.end_date or args.start_date),
                                     args.version, descriptor=args.descriptor, repointing=repointing_number)
    if args.instrument == 'swapi' and (args.data_level == 'l3a' or args.data_level == 'l3b'):
        processor = SwapiProcessor(processing_input_collection, input_dependency)
        paths = processor.process()
    elif args.instrument == 'glows' and args.data_level in ['l3a', 'l3b']:
        processor = GlowsProcessor(processing_input_collection, input_dependency)
        paths = processor.process()
    elif args.instrument == 'swe' and args.data_level == 'l3':
        processor = SweProcessor(processing_input_collection, input_dependency)
        paths = processor.process()
    elif args.instrument == 'hit' and args.data_level == 'l3':
        processor = HitProcessor(processing_input_collection, input_dependency)
        paths = processor.process()
    elif args.instrument == 'hi' and args.data_level == 'l3':
        if args.descriptor == "all-maps":
            initializer = HiL3Initializer()
            paths = []
            for map_descriptor in HI_SP_MAP_DESCRIPTORS:
                maps_to_produce: list[PossibleMapToProduce] = initializer.get_maps_that_should_be_produced(
                    map_descriptor)
                for dependency in maps_to_produce:
                    initializer.furnish_spice_dependencies(dependency)
                    processor = HiProcessor(dependency.processing_input_collection, dependency.input_metadata)
                    paths.extend(processor.process())
        else:
            processor = HiProcessor(processing_input_collection, input_dependency)
            paths = processor.process()
    elif args.instrument == 'ultra' and args.data_level == 'l3':
        processor = UltraProcessor(processing_input_collection, input_dependency)
        paths = processor.process()
    elif args.instrument == 'lo' and args.data_level == 'l3':
        processor = LoProcessor(processing_input_collection, input_dependency)
        paths = processor.process()
    elif args.instrument == 'codice':
        if args.descriptor.startswith("hi") and args.data_level in ['l3a', 'l3b']:
            processor = CodiceHiProcessor(processing_input_collection, input_dependency)
            paths = processor.process()
        elif args.descriptor.startswith("lo") and args.data_level == 'l3a':
            processor = CodiceLoProcessor(processing_input_collection, input_dependency)
            paths = processor.process()
        else:
            raise NotImplementedError(f"Unknown descriptor '{args.descriptor}' for codice instrument")
    else:
        raise NotImplementedError(
            f'Level {args.data_level} data processing has not yet been implemented for {args.instrument}')

    if args.upload_to_sdc:
        exceptions = []
        for path in paths:
            try:
                imap_data_access.upload(path)
            except Exception as e:
                exceptions.append(e)
        if exceptions:
            raise IOError(f"Failed to upload some files: {exceptions}")


def _furnish_spice_kernels(processing_input_collection):
    spice_kernel_paths = processing_input_collection.get_file_paths(data_type='spice')
    for kernel in spice_kernel_paths:
        kernel_path = imap_data_access.download(kernel)
        spiceypy.furnsh(str(kernel_path))


if __name__ == '__main__':
    with TemporaryDirectory() as dir:
        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        try:
            imap_l3_processor()
        except Exception as e:
            logger.error("Unhandled Exception:")
            raise e
