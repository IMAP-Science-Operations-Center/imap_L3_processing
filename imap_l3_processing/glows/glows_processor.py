import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import run
from typing import Optional
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath, AncillaryFilePath
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.descriptors import GLOWS_L3A_DESCRIPTOR
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.glows_toolkit.l3a_data import L3aData
from imap_l3_processing.glows.l3a.models import GlowsL3LightCurve
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3bc.cannot_process_carrington_rotation_error import CannotProcessCarringtonRotationError
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.glows_l3bc_initializer import GlowsL3BCInitializer, GlowsL3BCInitializerData
from imap_l3_processing.glows.l3bc.models import GlowsL3BIonizationRate, GlowsL3CSolarWind, GlowsL3BCProcessorOutput, \
    ExternalDependencies, read_pipeline_settings
from imap_l3_processing.glows.l3bc.science.filter_out_bad_days import filter_l3a_files
from imap_l3_processing.glows.l3bc.science.generate_l3bc import generate_l3bc
from imap_l3_processing.glows.l3bc.utils import get_pointing_date_range
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.glows_l3d_initializer import GlowsL3DInitializer
from imap_l3_processing.glows.l3d.models import GlowsL3DProcessorOutput
from imap_l3_processing.glows.l3d.utils import create_glows_l3b_json_file_from_cdf, create_glows_l3c_json_file_from_cdf, \
    PATH_TO_L3D_TOOLKIT, convert_json_to_l3d_data_product, get_parent_file_names_from_l3d_json, rename_l3d_text_outputs
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.glows.l3e.glows_l3e_initializer import GlowsL3EInitializer, GlowsL3EInitializerOutput
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.glows.l3e.glows_l3e_utils import determine_call_args_for_l3e_executable
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data

logger = logging.getLogger(__name__)


class GlowsProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = GlowsL3ADependencies.fetch_dependencies(self.dependencies)
            self.input_metadata.repointing = l3a_dependencies.repointing
            l3a_output = self.process_l3a(l3a_dependencies)
            l3a_output.parent_file_names = self.get_parent_file_names()
            cdf = save_data(l3a_output)
            return [cdf]
        elif self.input_metadata.data_level == "l3b":
            products_list = []

            l3bc_initializer_data: GlowsL3BCInitializerData = GlowsL3BCInitializer.get_crs_to_process(self.dependencies)

            crs_to_process_infos = [f"{dep.carrington_rotation_number} with version {dep.version}" for dep in l3bc_initializer_data.l3bc_dependencies]

            if len(crs_to_process_infos) > 0:
                logger.info("Found CRs to Process L3BC:")
                for dep in l3bc_initializer_data.l3bc_dependencies:
                    l3a_file_names = [l3a_d["filename"] for l3a_d in dep.l3a_data]
                    logger.info(f"\t{dep.carrington_rotation_number}, v{dep.version:03}: {l3a_file_names}")
            else:
                logger.info("No CRs to process for B/C")

            glows_l3bc_output_data = process_l3bc(self, l3bc_initializer_data)
            products_list.extend(glows_l3bc_output_data.data_products)

            l3bs = list({**l3bc_initializer_data.l3bs_by_cr, **glows_l3bc_output_data.l3bs_by_cr}.values())
            l3cs = list({**l3bc_initializer_data.l3cs_by_cr, **glows_l3bc_output_data.l3cs_by_cr}.values())
            l3d_initializer_result = GlowsL3DInitializer.should_process_l3d(l3bc_initializer_data.external_dependencies,
                                                                            l3bs, l3cs)
            if l3d_initializer_result is None:
                return products_list

            version_number, glows_l3d_dependency, old_l3d = l3d_initializer_result

            logger.info(f"Processing L3d with version: {version_number}")

            process_l3d_result = process_l3d(glows_l3d_dependency, version_number)

            if process_l3d_result is not None:
                logger.info(f"Finished processing L3d up to CR: {process_l3d_result.last_processed_cr}")
                products_list.extend([*process_l3d_result.l3d_text_file_paths, process_l3d_result.l3d_cdf_file_path])

                logger.info(f"Saved L3d CDF output to: {process_l3d_result.l3d_cdf_file_path}")
                for txt_file in process_l3d_result.l3d_text_file_paths:
                    logger.info(f"Saved L3d text file output to: {txt_file}")
            else:
                return products_list

            l3e_initializer_output = GlowsL3EInitializer.get_repointings_to_process(process_l3d_result, old_l3d, l3bc_initializer_data.repoint_file_path)

            logger.info(f"Processing L3e for repointings: {l3e_initializer_output.repointings.repointing_numbers}")

            products_list.extend(process_l3e(l3e_initializer_output))

            return products_list

    def process_l3a(self, dependencies: GlowsL3ADependencies) -> GlowsL3LightCurve:
        data = dependencies.data
        l3_data = L3aData(dependencies.ancillary_files)
        l3_data.process_l2_data_file(data)
        l3_data.generate_l3a_data(dependencies.ancillary_files)
        data_with_spin_angle = self.add_spin_angle_delta(l3_data.data, dependencies.ancillary_files)

        return create_glows_l3a_from_dictionary(data_with_spin_angle,
                                                replace(self.input_metadata, descriptor=GLOWS_L3A_DESCRIPTOR))

    @staticmethod
    def add_spin_angle_delta(data: dict, ancillary_files: dict) -> dict:
        with open(ancillary_files['settings']) as f:
            settings_file = json.load(f)
            number_of_bins = settings_file['l3a_nominal_number_of_bins']

        delta = 360 / number_of_bins / 2
        data['daily_lightcurve']['spin_angle_delta'] = np.full_like(data['daily_lightcurve']['spin_angle'], delta)

        return data

    @staticmethod
    def archive_dependencies(l3bc_deps: GlowsL3BCDependencies, external_dependencies: ExternalDependencies) -> Path:
        start_date = l3bc_deps.start_date.strftime("%Y%m%d")
        zip_path = TEMP_CDF_FOLDER_PATH / f"imap_glows_l3b-archive_{start_date}_v{l3bc_deps.version:03}.zip"
        json_filename = "cr_to_process.json"
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as file:
            file.write(external_dependencies.lyman_alpha_path, "lyman_alpha_composite.nc")
            file.write(external_dependencies.omni2_data_path, "omni2_all_years.dat")
            file.write(external_dependencies.f107_index_file_path, "f107_fluxtable.txt")
            cr = {"cr_rotation_number": l3bc_deps.carrington_rotation_number,
                  "l3a_paths": [l3a['filename'] for l3a in l3bc_deps.l3a_data],
                  "cr_start_date": str(l3bc_deps.start_date),
                  "cr_end_date": str(l3bc_deps.end_date),
                  "bad_days_list": l3bc_deps.ancillary_files['bad_days_list'].name,
                  "pipeline_settings": l3bc_deps.ancillary_files['pipeline_settings'].name,
                  "waw_helioion_mp": l3bc_deps.ancillary_files['WawHelioIonMP_parameters'].name,
                  "uv_anisotropy": l3bc_deps.ancillary_files['uv_anisotropy'].name,
                  "repointing_file": l3bc_deps.repointing_file_path.name,
                  }
            json_string = json.dumps(cr)
            file.writestr(json_filename, json_string)
        return zip_path

def process_l3bc(processor, initializer_data: GlowsL3BCInitializerData):
    l3bs_by_cr = {}
    l3cs_by_cr = {}
    data_products = []
    for dependency in initializer_data.l3bc_dependencies:
        logger.info(f"Processing L3BC for CR: {dependency.carrington_rotation_number}")
        zip_path = GlowsProcessor.archive_dependencies(l3bc_deps=dependency,
                                                       external_dependencies=initializer_data.external_dependencies)
        logger.info(f"Archived L3BC Dependencies to {zip_path}")

        filtered_days = filter_l3a_files(dependency.l3a_data, dependency.ancillary_files['bad_days_list'],
                                         dependency.carrington_rotation_number)
        try:
            l3b_data, l3c_data = generate_l3bc(replace(dependency, l3a_data=filtered_days))
        except CannotProcessCarringtonRotationError as e:
            print(f"skipping CR {dependency.carrington_rotation_number}:", e)
            continue

        l3b_metadata = InputMetadata("glows", "l3b", dependency.start_date, dependency.end_date,
                                     f"v{dependency.version:03}", "ion-rate-profile")
        l3c_metadata = InputMetadata("glows", "l3c", dependency.start_date, dependency.end_date,
                                     f"v{dependency.version:03}", "sw-profile")

        l3b_data_product = GlowsL3BIonizationRate.from_instrument_team_dictionary(l3b_data, l3b_metadata)
        l3c_data_product = GlowsL3CSolarWind.from_instrument_team_dictionary(l3c_data, l3c_metadata)

        l3b_data_product.parent_file_names += processor.get_parent_file_names([zip_path])
        l3b_cdf = save_data(l3b_data_product, cr_number=dependency.carrington_rotation_number)

        l3c_data_product.parent_file_names += processor.get_parent_file_names([zip_path, Path(l3b_cdf)])
        l3c_cdf = save_data(l3c_data_product, cr_number=dependency.carrington_rotation_number)

        logger.info(f"Finished processing CR: {dependency.carrington_rotation_number}")
        for path in [l3b_cdf, l3c_cdf]:
            logger.info(f"Saved {path}")

        l3bs_by_cr[dependency.carrington_rotation_number] = l3b_cdf.name
        l3cs_by_cr[dependency.carrington_rotation_number] = l3c_cdf.name
        data_products.extend([l3b_cdf, l3c_cdf, zip_path])

    return GlowsL3BCProcessorOutput(
        l3bs_by_cr=l3bs_by_cr,
        l3cs_by_cr=l3cs_by_cr,
        data_products=data_products
    )

def process_l3d(dependencies: GlowsL3DDependencies, version: int) -> Optional[GlowsL3DProcessorOutput]:

    [create_glows_l3b_json_file_from_cdf(l3b) for l3b in dependencies.l3b_file_paths]
    [create_glows_l3c_json_file_from_cdf(l3c) for l3c in dependencies.l3c_file_paths]

    os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3d', exist_ok=True)
    os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt', exist_ok=True)

    cr_to_process = read_pipeline_settings(dependencies.ancillary_files['pipeline_settings'])["start_cr"]

    file_manifest = {
        'external_files': {key: str(val) for key, val in dependencies.external_files.items()},
        'ancillary_files': {
            'pipeline_settings': str(dependencies.ancillary_files['pipeline_settings']),
            'WawHelioIon': {key: str(val) for key, val in dependencies.ancillary_files['WawHelioIon'].items()}
        },
    }

    last_processed_cr = None
    try:
        while True:
            output: subprocess.CompletedProcess = run(
                [sys.executable, './generate_l3d.py', f'{cr_to_process}', json.dumps(file_manifest)],
                cwd=str(PATH_TO_L3D_TOOLKIT),
                check=True,
                capture_output=True, text=True)
            if output.stdout:
                last_processed_cr = int(output.stdout.split('= ')[-1])

            cr_to_process += 1
    except subprocess.CalledProcessError as e:
        if 'L3d not generated: there is not enough L3b data to interpolate' not in e.stderr:
            raise Exception(e.stderr) from e

    if last_processed_cr:
        formatted_version = f"v{version:03}"

        output_text_files = []
        for text_file in os.listdir(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt'):
            if str(last_processed_cr) in text_file:
                output_text_files.append(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / text_file)

        txt_files_with_correct_version = rename_l3d_text_outputs(output_text_files, formatted_version)

        for txt_file in txt_files_with_correct_version:
            shutil.copy(txt_file, generate_imap_file_path(txt_file.name).construct_path())

        file_name = f'imap_glows_l3d_solar-params-history_19470303-cr0{last_processed_cr}_v00.json'

        start_date = datetime(1947, 3, 3)
        data_product_metadata = InputMetadata(instrument="glows", data_level="l3d", descriptor="solar-hist",
                                              start_date=start_date, end_date=start_date, version=formatted_version)
        parent_file_names = get_parent_file_names_from_l3d_json(PATH_TO_L3D_TOOLKIT / 'data_l3d')

        l3d_data_product = convert_json_to_l3d_data_product(PATH_TO_L3D_TOOLKIT / 'data_l3d' / file_name,
                                                            data_product_metadata, parent_file_names)
        l3d_data_product_path = save_data(l3d_data_product, cr_number=last_processed_cr)

        return GlowsL3DProcessorOutput(l3d_data_product_path, txt_files_with_correct_version, last_processed_cr)
    return None

def process_l3e_lo(
        parent_file_names: list[str],
        repointing: int,
        repointing_start: datetime,
        epoch_delta: timedelta,
        elongation_value: int,
        version: int
) -> list[Path]:
    repointing_midpoint = repointing_start + epoch_delta
    l3e_args = determine_call_args_for_l3e_executable(repointing_start, repointing_midpoint, float(elongation_value))
    call_args = l3e_args.to_argument_list()

    logger.info(f"Processing L3e Lo, calling survProbLo with {call_args}")

    run(["./survProbLo"] + call_args)

    input_metadata = InputMetadata(
        instrument="glows",
        data_level="l3e",
        descriptor="survival-probability-lo",
        start_date=repointing_start,
        end_date=repointing_start + epoch_delta * 2,
        version=f"v{version:03}",
        repointing=repointing,
    )

    elongation_in_filename = f"{elongation_value}."
    elongation_in_filename += "0" * (5-len(elongation_in_filename))

    output_path = Path(f'probSur.Imap.Lo_{l3e_args.formatted_date}_{l3e_args.decimal_date[:8]}_{elongation_in_filename}.dat')
    lo_data = GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product(input_metadata, output_path,
                                                                 repointing_midpoint, elongation_value, l3e_args)

    lo_data.parent_file_names = parent_file_names

    lo_cdf = save_data(lo_data)

    cdf_science_file_path = ScienceFilePath(lo_cdf)
    new_dat_path = AncillaryFilePath.generate_from_inputs(
        instrument="glows",
        descriptor=f"{cdf_science_file_path.descriptor}-raw",
        start_time=cdf_science_file_path.start_date,
        version=cdf_science_file_path.version,
        extension="dat"
    ).construct_path()

    shutil.move(output_path, new_dat_path)

    return [lo_cdf, new_dat_path]

def process_l3e_ul(parent_file_names: list[str], repointing: int, repointing_start: datetime, epoch_delta: timedelta, version: int) -> list[Path]:
    repointing_midpoint = repointing_start + epoch_delta
    call_args_object = determine_call_args_for_l3e_executable(repointing_start, repointing_midpoint, 30)
    call_args = call_args_object.to_argument_list()

    logger.info(f"Processing L3e Ultra, calling survProbUltra with {call_args}")

    run(["./survProbUltra"] + call_args)

    input_metadata = InputMetadata(
        instrument="glows",
        data_level="l3e",
        descriptor="survival-probability-ul",
        start_date=repointing_start,
        end_date=repointing_start + epoch_delta * 2,
        version=f"v{version:03}",
        repointing=repointing,
    )

    output_path = Path(f'probSur.Imap.Ul_{call_args[0]}_{call_args[1][:8]}.dat')
    ul_data = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(input_metadata, output_path,
                                                                    repointing_midpoint, call_args_object)

    ul_data.parent_file_names = parent_file_names

    ul_cdf = save_data(ul_data)

    cdf_science_file_path = ScienceFilePath(ul_cdf)

    new_dat_path = AncillaryFilePath.generate_from_inputs(
        instrument="glows",
        descriptor=f"{cdf_science_file_path.descriptor}-raw",
        start_time=cdf_science_file_path.start_date,
        version=cdf_science_file_path.version,
        extension="dat"
    ).construct_path()

    shutil.move(output_path, new_dat_path)

    return [ul_cdf, new_dat_path]

def process_l3e_hi(parent_file_names: list[str], repointing: int, repointing_start: datetime, epoch_delta: timedelta, elongation: int, version: int) -> list[Path]:
    repointing_midpoint = repointing_start + epoch_delta
    l3e_hi_args = determine_call_args_for_l3e_executable(repointing_start, repointing_midpoint, elongation)
    call_args = l3e_hi_args.to_argument_list()

    logger.info(f"Processing L3e Hi, calling survProbHi with {call_args}")

    run(["./survProbHi"] + call_args)

    input_metadata = InputMetadata(instrument='glows', descriptor=f'survival-probability-hi-{180-elongation}',
                                   version=f'v{version:03}', start_date=repointing_start, end_date=repointing_start + epoch_delta * 2,
                                   repointing=repointing, data_level='l3e')

    output_path = Path(f'probSur.Imap.Hi_{call_args[0]}_{call_args[1][:8]}_{call_args[-1][:5]}.dat')

    hi_data = GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product(
        input_metadata,
        output_path,
        repointing_midpoint,
        l3e_hi_args
    )
    hi_data.parent_file_names = parent_file_names

    hi_cdf = save_data(hi_data)

    cdf_science_file_path = ScienceFilePath(hi_cdf)
    new_dat_path = AncillaryFilePath.generate_from_inputs(
        instrument="glows",
        descriptor=f"{cdf_science_file_path.descriptor}-raw",
        start_time=cdf_science_file_path.start_date,
        version=cdf_science_file_path.version,
        extension="dat"
    ).construct_path()

    shutil.move(output_path, new_dat_path)

    return [hi_cdf, new_dat_path]

def process_l3e(initializer_data: GlowsL3EInitializerOutput):
    products_list = []

    for repointing in initializer_data.repointings.repointing_numbers:
        with SwallowExceptionAndLog(f"Exception encountered when processing L3e for repointing {repointing}"):
            start_repointing, end_repointing = get_pointing_date_range(repointing)
            epoch_delta: timedelta = (end_repointing - start_repointing) / 2

            with SwallowExceptionAndLog(f"Exception encountered when processing L3e lo for repointing {repointing}"):
                lo_parent_file_names = initializer_data.dependencies.get_lo_parents()
                repointing_doy = start_repointing.timetuple().tm_yday
                lo_elongation = initializer_data.dependencies.elongation.get(f"{start_repointing.year}{repointing_doy:03}")
                if lo_elongation is not None:
                    lo_version = initializer_data.repointings.lo_repointings[repointing]
                    products_list.extend(process_l3e_lo(lo_parent_file_names, repointing, start_repointing, epoch_delta, lo_elongation, lo_version))
                else:
                    logger.warning(f"Skipping L3e Lo processing for {repointing=} because there is no elongation data")

            with SwallowExceptionAndLog(f"Exception encountered when processing L3e hi-90 for repointing {repointing}"):
                hi_parent_file_names = initializer_data.dependencies.get_hi_parents()
                hi_90_version = initializer_data.repointings.hi_90_repointings[repointing]
                products_list.extend(process_l3e_hi(hi_parent_file_names, repointing, start_repointing, epoch_delta, 90, hi_90_version))

            with SwallowExceptionAndLog(f"Exception encountered when processing L3e hi-45 for repointing {repointing}"):
                hi_parent_file_names = initializer_data.dependencies.get_hi_parents()
                hi_45_version = initializer_data.repointings.hi_45_repointings[repointing]
                products_list.extend(process_l3e_hi(hi_parent_file_names, repointing, start_repointing, epoch_delta, 135, hi_45_version))

            with SwallowExceptionAndLog(f"Exception encountered when processing L3e ultra for repointing {repointing}"):
                ul_parent_file_names = initializer_data.dependencies.get_ul_parents()
                ul_version = initializer_data.repointings.ultra_repointings[repointing]
                products_list.extend(process_l3e_ul(ul_parent_file_names, repointing, start_repointing, epoch_delta, ul_version))

    return products_list

class SwallowExceptionAndLog:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is KeyboardInterrupt:
            return False
        elif exc_type is not None:
            print(self.message)
            traceback.print_exception(exc_type, exc_val, exc_tb)
        return True