import json
from copy import copy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import run

import imap_data_access
import numpy as np

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
from imap_l3_processing.glows.l3bc.utils import make_l3b_data_with_fill, make_l3c_data_with_fill, get_repoint_date_range
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.glows.l3e.glows_l3e_utils import determine_call_args_for_l3e_executable
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class GlowsProcessor(Processor):

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = GlowsL3ADependencies.fetch_dependencies(self.dependencies)
            self.input_metadata.repointing = l3a_dependencies.repointing
            l3a_output = self.process_l3a(l3a_dependencies)
            l3a_output.parent_file_names = self.get_parent_file_names()
            proton_cdf = save_data(l3a_output)
            imap_data_access.upload(proton_cdf)
        elif self.input_metadata.data_level == "l3b":
            zip_files = GlowsInitializer.validate_and_initialize(self.input_metadata.version)
            for zip_file in zip_files:
                dependencies = GlowsL3BCDependencies.fetch_dependencies(zip_file)
                l3b_data_product, l3c_data_product = self.process_l3bc(dependencies)
                l3b_cdf = save_data(l3b_data_product)
                l3c_data_product.parent_file_names.append(Path(l3b_cdf).name)
                l3c_cdf = save_data(l3c_data_product)
                imap_data_access.upload(l3b_cdf)
                imap_data_access.upload(l3c_cdf)
                imap_data_access.upload(zip_file)
        elif self.input_metadata.data_level == "l3e":
            l3e_dependencies, repointing_number = GlowsL3EDependencies.fetch_dependencies(self.dependencies,
                                                                                          self.input_metadata.descriptor)
            epoch, epoch_end = get_repoint_date_range(repointing_number)
            epoch = epoch.astype(datetime)
            epoch_end = epoch_end.astype(datetime)
            epoch_delta = (epoch_end - epoch) / 2

            l3e_dependencies.rename_dependencies()

            if self.input_metadata.descriptor == "survival-probability-lo":
                self.process_l3e_lo(epoch, epoch_delta)
            elif self.input_metadata.descriptor == "survival-probability-hi-45":
                self.process_l3e_hi(epoch, epoch_delta, 135)
            elif self.input_metadata.descriptor == "survival-probability-hi-90":
                self.process_l3e_hi(epoch, epoch_delta, 90)
            elif self.input_metadata.descriptor == "survival-probability-ul":
                self.process_l3e_ul(epoch, epoch_delta)

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
        l3b_metadata = UpstreamDataDependency("glows", "l3b", dependencies.start_date, dependencies.end_date,
                                              self.input_metadata.version, "ion-rate-profile")
        l3c_metadata = UpstreamDataDependency("glows", "l3c", dependencies.start_date, dependencies.end_date,
                                              self.input_metadata.version, "sw-profile")

        try:
            l3b_data, l3c_data = generate_l3bc(replace(dependencies, l3a_data=filtered_days))
            l3b_data_product = GlowsL3BIonizationRate.from_instrument_team_dictionary(l3b_data,
                                                                                      l3b_metadata)
            l3c_data_product = GlowsL3CSolarWind.from_instrument_team_dictionary(l3c_data, l3c_metadata)
        except CannotProcessCarringtonRotationError:
            l3b_data_with_fills = make_l3b_data_with_fill(dependencies)
            l3c_data_with_fills = make_l3c_data_with_fill(dependencies)
            l3b_data_product = GlowsL3BIonizationRate.from_instrument_team_dictionary(l3b_data_with_fills,
                                                                                      l3b_metadata)
            l3c_data_product = GlowsL3CSolarWind.from_instrument_team_dictionary(l3c_data_with_fills,
                                                                                 l3c_metadata)
        l3b_data_product.parent_file_names.append(dependencies.zip_file_path.name)
        l3c_data_product.parent_file_names.append(dependencies.zip_file_path.name)
        return l3b_data_product, l3c_data_product

    def process_l3e_lo(self, epoch: datetime, epoch_delta: timedelta):
        call_args = determine_call_args_for_l3e_executable(epoch, epoch + epoch_delta, 90)

        run(["./survProbLo"] + call_args)

        output_path = Path(f'probSur.Imap.Lo_{call_args[0]}_{call_args[1][:8]}_{call_args[-1][:5]}.dat')
        lo_data = GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product(self.input_metadata, output_path,
                                                                     np.array([epoch]), np.array([epoch_delta]))

        lo_cdf = save_data(lo_data)
        imap_data_access.upload(lo_cdf)

    def process_l3e_hi(self, epoch: datetime, epoch_delta: timedelta, elongation: int):
        call_args = determine_call_args_for_l3e_executable(epoch, epoch + epoch_delta, elongation)
        run(["./survProbHi"] + call_args)

        output_path = Path(f'probSur.Imap.Hi_{call_args[0]}_{call_args[1][:8]}_{call_args[-1][:5]}.dat')
        hi_data = GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product(self.input_metadata, output_path,
                                                                     np.array([epoch]), np.array([epoch_delta]))

        hi_cdf = save_data(hi_data)
        imap_data_access.upload(hi_cdf)

    def process_l3e_ul(self, epoch: datetime, epoch_delta: timedelta):
        call_args = determine_call_args_for_l3e_executable(epoch, epoch + epoch_delta, 30)

        run(["./survProbUltra"] + call_args)

        output_path = Path(f'probSur.Imap.Ul_{call_args[0]}_{call_args[1][:8]}.dat')
        ul_data = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(self.input_metadata, output_path,
                                                                        np.array([epoch]), np.array([epoch_delta]))

        ul_cdf = save_data(ul_data)
        imap_data_access.upload(ul_cdf)

    def process_l3e(self):
        l3e_dependencies, repointing = GlowsL3EDependencies.fetch_dependencies(self.dependencies)

        l3e_dependencies.rename_dependencies()

        repointing_start_date, repointing_end_date = get_repoint_date_range(repointing)

        repointing_start_date = repointing_start_date.astype(datetime)
        repointing_end_date = repointing_end_date.astype(datetime)

        epoch_delta = (repointing_end_date - repointing_start_date) / 2
        mid_point = repointing_start_date + epoch_delta

        lo_call_args = determine_call_args_for_l3e_executable(repointing_start_date
                                                              , mid_point, 90)
        hi90_call_args = determine_call_args_for_l3e_executable(repointing_start_date,
                                                                mid_point, 90)
        hi45_call_args = determine_call_args_for_l3e_executable(repointing_start_date,
                                                                mid_point, 135)
        ultra_call_args = determine_call_args_for_l3e_executable(repointing_start_date,
                                                                 mid_point, 30)

        lo_call_args_array = [arg for arg in lo_call_args.split(' ')]
        hi90_call_args_array = [arg for arg in hi90_call_args.split(' ')]
        hi45_call_args_array = [arg for arg in hi45_call_args.split(' ')]
        ultra_call_args_array = [arg for arg in ultra_call_args.split(' ')]

        run(['./survProbLo'] + lo_call_args_array)
        run(['./survProbHi'] + hi90_call_args_array)
        run(['./survProbHi'] + hi45_call_args_array)
        run(['./survProbUltra'] + ultra_call_args_array)

        lo_data_path = Path(
            f'probSur.Imap.Lo_{lo_call_args_array[0]}_{lo_call_args_array[1][:8]}_{lo_call_args_array[-1][:5]}.dat')

        lo_input_metadata = copy(self.input_metadata)
        lo_input_metadata.descriptor = 'survival-probability-lo'
        lo_data = GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product(lo_input_metadata,
                                                                     lo_data_path,
                                                                     np.array([repointing_start_date]),
                                                                     np.array([epoch_delta]))
        hi_45_path = Path(
            f'probSur.Imap.Hi_{hi45_call_args_array[0]}_{hi45_call_args_array[1][:8]}_{hi45_call_args_array[-1][:5]}.dat')

        hi45_input_metadata = copy(self.input_metadata)
        hi45_input_metadata.descriptor = 'survival-probability-hi-45'
        hi_45_data = GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product(hi45_input_metadata,
                                                                        hi_45_path,
                                                                        np.array([repointing_start_date]),
                                                                        np.array([epoch_delta]))

        hi_90_path = Path(
            f'probSur.Imap.Hi_{hi90_call_args_array[0]}_{hi90_call_args_array[1][:8]}_{hi90_call_args_array[-1][:5]}.dat')

        hi90_input_metadata = copy(self.input_metadata)
        hi90_input_metadata.descriptor = 'survival-probability-hi-90'

        hi_90_data = GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product(hi90_input_metadata,
                                                                        hi_90_path,
                                                                        np.array([repointing_start_date]),
                                                                        np.array([epoch_delta]))

        ultra_path = Path(
            f'probSur.Imap.Ul_{ultra_call_args_array[0]}_{ultra_call_args_array[1][:8]}.dat')

        ultra_input_metadata = copy(self.input_metadata)
        ultra_input_metadata.descriptor = 'survival-probability-ul'
        ultra_data = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(ultra_input_metadata,
                                                                           ultra_path,
                                                                           np.array([repointing_start_date]),
                                                                           np.array([epoch_delta]))

        return lo_data, hi_45_data, hi_90_data, ultra_data

    @staticmethod
    def add_spin_angle_delta(data: dict, ancillary_files: dict) -> dict:
        with open(ancillary_files['settings']) as f:
            settings_file = json.load(f)
            number_of_bins = settings_file['l3a_nominal_number_of_bins']

        delta = 360 / number_of_bins / 2
        data['daily_lightcurve']['spin_angle_delta'] = np.full_like(data['daily_lightcurve']['spin_angle'], delta)

        return data
