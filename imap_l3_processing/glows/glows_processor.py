import json

import imap_data_access
import numpy as np

from imap_l3_processing.glows.glows_initializer import GlowsInitializer
from imap_l3_processing.glows.glows_toolkit.l3a_data import L3aData
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.models import GlowsL3LightCurve
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3b.glows_l3b_dependencies import GlowsInitializerDependencies
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class GlowsProcessor(Processor):

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = GlowsL3ADependencies.fetch_dependencies(self.dependencies)
            l3a_output = self.process_l3a(l3a_dependencies)
            proton_cdf = save_data(l3a_output)
            imap_data_access.upload(proton_cdf)
        elif self.input_metadata.data_level == "l3b":
            l3b_dependencies = GlowsInitializerDependencies.fetch_dependencies(self.dependencies)
            initializer = GlowsInitializer()
            if initializer.should_process(l3b_dependencies):
                save_data(None)
                imap_data_access.upload("")

    def process_l3a(self, dependencies: GlowsL3ADependencies) -> GlowsL3LightCurve:
        data = dependencies.data
        l3_data = L3aData(dependencies.ancillary_files)
        l3_data.process_l2_data_file(data)
        l3_data.generate_l3a_data(dependencies.ancillary_files)
        data_with_spin_angle = self.add_spin_angle_delta(l3_data.data, dependencies.ancillary_files)

        return create_glows_l3a_from_dictionary(data_with_spin_angle, self.input_metadata.to_upstream_data_dependency(
            self.dependencies[0].descriptor))

    @staticmethod
    def add_spin_angle_delta(data: dict, ancillary_files: dict) -> dict:
        with open(ancillary_files['settings']) as f:
            settings_file = json.load(f)
            number_of_bins = settings_file['l3a_nominal_number_of_bins']

        delta = 360 / number_of_bins / 2
        data['daily_lightcurve']['spin_angle_delta'] = np.full_like(data['daily_lightcurve']['spin_angle'], delta)

        return data
