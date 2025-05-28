import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import run

import imap_data_access
import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.glows.descriptors import GLOWS_L3A_DESCRIPTOR
from imap_l3_processing.glows.glows_initializer import GlowsInitializer
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.glows_toolkit.l3a_data import L3aData
from imap_l3_processing.glows.l3a.models import GlowsL3LightCurve
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3bc.cannot_process_carrington_rotation_error import CannotProcessCarringtonRotationError
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import GlowsL3BIonizationRate, GlowsL3CSolarWind
from imap_l3_processing.glows.l3bc.science.filter_out_bad_days import filter_out_bad_days
from imap_l3_processing.glows.l3bc.science.generate_l3bc import generate_l3bc
from imap_l3_processing.glows.l3bc.utils import get_pointing_date_range
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.utils import create_glows_l3b_json_file_from_cdf, create_glows_l3c_json_file_from_cdf, \
    PATH_TO_L3D_TOOLKIT, convert_json_to_l3d_data_product, get_parent_file_names_from_l3d_json, set_version_on_txt_files
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.glows.l3e.glows_l3e_utils import determine_call_args_for_l3e_executable, \
    determine_l3e_files_to_produce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class GlowsProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = GlowsL3ADependencies.fetch_dependencies(self.dependencies)
            self.input_metadata.repointing = l3a_dependencies.repointing
            l3a_output = self.process_l3a(l3a_dependencies)
            l3a_output.parent_file_names = self.get_parent_file_names()
            proton_cdf = save_data(l3a_output)
            imap_data_access.upload(proton_cdf)
        elif self.input_metadata.data_level == "l3b":
            zip_files = GlowsInitializer.validate_and_initialize(self.input_metadata.version, self.dependencies)
            for zip_file in zip_files:
                dependencies = GlowsL3BCDependencies.fetch_dependencies(zip_file)
                l3b_data_product, l3c_data_product = self.process_l3bc(dependencies)
                l3b_cdf = save_data(l3b_data_product, cr_number=dependencies.carrington_rotation_number)
                l3c_data_product.parent_file_names.append(Path(l3b_cdf).name)
                l3c_cdf = save_data(l3c_data_product, cr_number=dependencies.carrington_rotation_number)
                imap_data_access.upload(l3b_cdf)
                imap_data_access.upload(l3c_cdf)
                imap_data_access.upload(zip_file)
        elif self.input_metadata.data_level == "l3d":
            l3d_dependencies = GlowsL3DDependencies.fetch_dependencies(self.dependencies)
            data_product, l3d_txt_paths, last_processed_cr = self.process_l3d(l3d_dependencies)
            if data_product is not None and l3d_txt_paths is not None and last_processed_cr is not None:
                cdf = save_data(data_product, cr_number=last_processed_cr)
                imap_data_access.upload(cdf)
                for txt_path in l3d_txt_paths:
                    imap_data_access.upload(txt_path)
        elif self.input_metadata.data_level == "l3e":
            l3e_dependencies, cr_number = GlowsL3EDependencies.fetch_dependencies(self.dependencies,
                                                                                  self.input_metadata.descriptor)
            l3e_dependencies.rename_dependencies()

            repointings = determine_l3e_files_to_produce(self.input_metadata.descriptor,
                                                         l3e_dependencies.pipeline_settings['start_cr'], cr_number,
                                                         self.input_metadata.version,
                                                         Path(os.getenv("REPOINT_DATA_FILEPATH")))
            for repointing in repointings:
                try:
                    epoch, epoch_end = get_pointing_date_range(repointing)
                    epoch_dt: datetime = epoch.astype('datetime64[us]').astype(datetime)
                    epoch_end_dt: datetime = epoch_end.astype('datetime64[us]').astype(datetime)
                    epoch_delta: timedelta = (epoch_end_dt - epoch_dt) / 2

                    if self.input_metadata.descriptor == "survival-probability-lo":
                        try:
                            year_with_repointing = str(epoch_dt.year) + str(int(repointing)).zfill(3)
                            elongation = l3e_dependencies.elongation[year_with_repointing]
                        except KeyError:
                            continue
                        self.process_l3e_lo(epoch_dt, epoch_delta, elongation)
                    elif self.input_metadata.descriptor == "survival-probability-hi-45":
                        self.process_l3e_hi(epoch_dt, epoch_delta, 135)
                    elif self.input_metadata.descriptor == "survival-probability-hi-90":
                        self.process_l3e_hi(epoch_dt, epoch_delta, 90)
                    elif self.input_metadata.descriptor == "survival-probability-ul":
                        self.process_l3e_ul(epoch_dt, epoch_delta)
                except Exception as e:
                    print("Exception encountered for repointing ", repointing, e)

    def process_l3a(self, dependencies: GlowsL3ADependencies) -> GlowsL3LightCurve:
        data = dependencies.data
        l3_data = L3aData(dependencies.ancillary_files)
        l3_data.process_l2_data_file(data)
        l3_data.generate_l3a_data(dependencies.ancillary_files)
        data_with_spin_angle = self.add_spin_angle_delta(l3_data.data, dependencies.ancillary_files)

        return create_glows_l3a_from_dictionary(data_with_spin_angle,
                                                replace(self.input_metadata, descriptor=GLOWS_L3A_DESCRIPTOR))

    def process_l3bc(self, dependencies: GlowsL3BCDependencies) -> tuple[GlowsL3BIonizationRate, GlowsL3CSolarWind]:
        filtered_days = filter_out_bad_days(dependencies.l3a_data, dependencies.ancillary_files['bad_days_list'])
        l3b_metadata = InputMetadata("glows", "l3b", dependencies.start_date, dependencies.end_date,
                                     self.input_metadata.version, "ion-rate-profile")
        l3c_metadata = InputMetadata("glows", "l3c", dependencies.start_date, dependencies.end_date,
                                     self.input_metadata.version, "sw-profile")

        try:
            l3b_data, l3c_data = generate_l3bc(replace(dependencies, l3a_data=filtered_days))
        except CannotProcessCarringtonRotationError as e:
            raise e

        l3b_data_product = GlowsL3BIonizationRate.from_instrument_team_dictionary(l3b_data,
                                                                                  l3b_metadata)
        l3c_data_product = GlowsL3CSolarWind.from_instrument_team_dictionary(l3c_data, l3c_metadata)

        l3b_data_product.parent_file_names += self.get_parent_file_names([dependencies.zip_file_path])
        l3c_data_product.parent_file_names += self.get_parent_file_names([dependencies.zip_file_path])
        return l3b_data_product, l3c_data_product

    def process_l3d(self, dependencies: GlowsL3DDependencies):
        [create_glows_l3b_json_file_from_cdf(l3b) for l3b in dependencies.l3b_file_paths]
        [create_glows_l3c_json_file_from_cdf(l3c) for l3c in dependencies.l3c_file_paths]

        os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3d', exist_ok=True)
        os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt', exist_ok=True)

        with open(dependencies.ancillary_files['pipeline_settings'], "r") as fp:
            pipeline_settings = json.load(fp)
            cr_to_process = int(pipeline_settings['start_cr'])

        file_manifest = {
            'ancillary_files': {
                'pipeline_settings': str(dependencies.ancillary_files['pipeline_settings']),
                'WawHelioIon': {key: str(val) for key, val in dependencies.ancillary_files['WawHelioIon'].items()}
            },
            'external_files': {key: str(val) for key, val in dependencies.external_files.items()}
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
            file_name = f'imap_glows_l3d_solar-params-history_19470303-cr0{last_processed_cr}_v00.json'

            parent_file_names = get_parent_file_names_from_l3d_json(PATH_TO_L3D_TOOLKIT / 'data_l3d')

            output_txt_files = [PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / last_cr_txt_file for last_cr_txt_file in
                                os.listdir(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt') if
                                str(last_processed_cr) in last_cr_txt_file]

            txt_files_with_correct_version = set_version_on_txt_files(output_txt_files, self.input_metadata.version)

            return convert_json_to_l3d_data_product(PATH_TO_L3D_TOOLKIT / 'data_l3d' / file_name, self.input_metadata,
                                                    parent_file_names), txt_files_with_correct_version, last_processed_cr
        return None, None, None

    def process_l3e_lo(self, epoch: datetime, epoch_delta: timedelta, elongation_value: int):
        call_args = determine_call_args_for_l3e_executable(epoch, epoch + epoch_delta, elongation_value)

        run(["./survProbLo"] + call_args)

        output_path = Path(f'probSur.Imap.Lo_{call_args[0]}_{call_args[1][:8]}_{call_args[-1][:5]}.dat')
        lo_data = GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product(self.input_metadata, output_path,
                                                                     np.array([epoch]), np.array([epoch_delta]),
                                                                     elongation_value)

        lo_data.parent_file_names = self.get_parent_file_names()
        lo_cdf = save_data(lo_data)
        imap_data_access.upload(lo_cdf)

    def process_l3e_hi(self, epoch: datetime, epoch_delta: timedelta, elongation: int):
        call_args = determine_call_args_for_l3e_executable(epoch, epoch + epoch_delta, elongation)
        run(["./survProbHi"] + call_args)

        output_path = Path(f'probSur.Imap.Hi_{call_args[0]}_{call_args[1][:8]}_{call_args[-1][:5]}.dat')
        hi_data = GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product(self.input_metadata, output_path,
                                                                     np.array([epoch]), np.array([epoch_delta]))

        hi_data.parent_file_names = self.get_parent_file_names()

        hi_cdf = save_data(hi_data)
        imap_data_access.upload(hi_cdf)

    def process_l3e_ul(self, epoch: datetime, epoch_delta: timedelta):
        call_args = determine_call_args_for_l3e_executable(epoch, epoch + epoch_delta, 30)

        run(["./survProbUltra"] + call_args)

        output_path = Path(f'probSur.Imap.Ul_{call_args[0]}_{call_args[1][:8]}.dat')
        ul_data = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(self.input_metadata, output_path,
                                                                        np.array([epoch]), np.array([epoch_delta]))

        ul_data.parent_file_names = self.get_parent_file_names()

        ul_cdf = save_data(ul_data)
        imap_data_access.upload(ul_cdf)

    @staticmethod
    def add_spin_angle_delta(data: dict, ancillary_files: dict) -> dict:
        with open(ancillary_files['settings']) as f:
            settings_file = json.load(f)
            number_of_bins = settings_file['l3a_nominal_number_of_bins']

        delta = 360 / number_of_bins / 2
        data['daily_lightcurve']['spin_angle_delta'] = np.full_like(data['daily_lightcurve']['spin_angle'], delta)

        return data
