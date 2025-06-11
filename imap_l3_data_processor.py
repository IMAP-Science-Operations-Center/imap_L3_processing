import argparse
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import imap_data_access
import spiceypy
from imap_data_access import ScienceInput, AncillaryInput, SPICEInput, RepointInput
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
    # processing_input_collection = ProcessingInputCollection()
    # processing_input_collection.deserialize(args.dependency)

    solar_hist = ScienceInput("imap_glows_l3d_solar-hist_19470303-cr02094_v004.cdf")
    p_dens = ScienceInput("imap_glows_l3d_p-dens_19470303-cr02094_v001.cdf")
    energy_grid_lo = AncillaryInput("imap_glows_energy-grid-lo_20100101_v002.dat")
    energy_grid_hi = AncillaryInput("imap_glows_energy-grid-hi_20100101_v002.dat")
    energy_grid_ultra = AncillaryInput("imap_glows_energy-grid-ultra_20100101_v002.dat")
    ionization_files = AncillaryInput("imap_glows_ionization-files_20100101_v002.dat")
    tess_xyz_8 = AncillaryInput("imap_glows_tess-xyz-8_20100101_v002.dat")
    tess_ang_16 = AncillaryInput("imap_glows_tess-ang-16_20100101_v002.dat")
    lya = ScienceInput("imap_glows_l3d_lya_19470303-cr02094_v001.cdf")
    phion = ScienceInput("imap_glows_l3d_phion_19470303-cr02094_v002.cdf")
    speed = ScienceInput("imap_glows_l3d_speed_19470303-cr02094_v001.cdf")
    uv_anis = ScienceInput("imap_glows_l3d_uv-anis_19470303-cr02094_v001.cdf")
    e_dens = ScienceInput("imap_glows_l3d_e-dens_19470303-cr02094_v001.cdf")
    pipeline_settings = AncillaryInput("imap_glows_pipeline-settings-l3bcde_19470303_v010.json")
    science_frames = SPICEInput("imap_science_0001.tf")
    ephemeris_reconstructed = SPICEInput("imap_recon_20250415_20260415_v01.bsp")
    de440 = SPICEInput("de440.bsp")
    attitude_history = SPICEInput("imap_2025_105_2026_105_02.ah.bc")
    pointing_attitude = SPICEInput("imap_dps_2025_105_2026_105_01.ah.bc")
    leapseconds = SPICEInput("naif0012.tls")
    spacecraft_clock = SPICEInput("imap_sclk_0000.tsc")
    repoint = RepointInput("imap_2026_269_05.repoint.csv")
    processing_input_collection = ProcessingInputCollection(solar_hist, p_dens, energy_grid_lo, energy_grid_hi,
                                                            energy_grid_ultra, ionization_files,
                                                            tess_xyz_8, tess_ang_16,
                                                            lya, phion, speed, uv_anis,
                                                            e_dens, pipeline_settings,
                                                            science_frames, ephemeris_reconstructed, de440,
                                                            attitude_history, leapseconds, spacecraft_clock,
                                                            repoint, pointing_attitude)
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

    processor.process()


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
