import json
from dataclasses import replace

import imap_data_access
import numpy as np

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
from imap_l3_processing.glows.l3bc.utils import make_l3b_data_with_fill, make_l3c_data_with_fill
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class GlowsProcessor(Processor):

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = GlowsL3ADependencies.fetch_dependencies(self.dependencies)
            self.input_metadata.repointing = l3a_dependencies.repointing
            l3a_output = self.process_l3a(l3a_dependencies)
            proton_cdf = save_data(l3a_output)
            imap_data_access.upload(proton_cdf)
        elif self.input_metadata.data_level == "l3b":
            zip_files = GlowsInitializer.validate_and_initialize(self.input_metadata.version)
            for zip_file in zip_files:
                imap_data_access.upload(zip_file)
                dependencies = GlowsL3BCDependencies.fetch_dependencies(zip_file)
                l3b_data_product, l3c_data_product = self.process_l3bc(dependencies)
                l3b_cdf = save_data(l3b_data_product)
                l3c_cdf = save_data(l3c_data_product)
                imap_data_access.upload(l3b_cdf)
                imap_data_access.upload(l3c_cdf)

    def process_l3a(self, dependencies: GlowsL3ADependencies) -> GlowsL3LightCurve:
        data = dependencies.data
        l3_data = L3aData(dependencies.ancillary_files)
        l3_data.process_l2_data_file(data)
        l3_data.generate_l3a_data(dependencies.ancillary_files)
        data_with_spin_angle = self.add_spin_angle_delta(l3_data.data, dependencies.ancillary_files)

        return create_glows_l3a_from_dictionary(data_with_spin_angle, self.input_metadata.to_upstream_data_dependency(
            self.dependencies[0].descriptor))

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
            l3c_data_with_fills = make_l3c_data_with_fill()
            l3b_data_product = GlowsL3BIonizationRate.from_instrument_team_dictionary(l3b_data_with_fills,
                                                                                      l3b_metadata)
            l3c_data_product = GlowsL3CSolarWind.from_instrument_team_dictionary(l3c_data_with_fills,
                                                                                 l3c_metadata)

        return l3b_data_product, l3c_data_product

    @staticmethod
    def add_spin_angle_delta(data: dict, ancillary_files: dict) -> dict:
        with open(ancillary_files['settings']) as f:
            settings_file = json.load(f)
            number_of_bins = settings_file['l3a_nominal_number_of_bins']

        delta = 360 / number_of_bins / 2
        data['daily_lightcurve']['spin_angle_delta'] = np.full_like(data['daily_lightcurve']['spin_angle'], delta)

        return data
