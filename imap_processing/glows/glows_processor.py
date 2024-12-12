import dataclasses

import imap_data_access

from imap_processing.glows.glows_toolkit.l3a_data import L3aData
from imap_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_processing.glows.l3a.models import GlowsL3LightCurve
from imap_processing.glows.l3a.utils import create_glows_l3a_from_dictionary
from imap_processing.processor import Processor
from imap_processing.utils import save_data


class GlowsProcessor(Processor):

    def process(self):
        dependencies = [
            dataclasses.replace(dep, start_date=self.input_metadata.start_date, end_date=self.input_metadata.end_date)
            for dep in
            self.dependencies]

        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = GlowsL3ADependencies.fetch_dependencies(dependencies)
            l3a_output = self.process_l3a(l3a_dependencies)
            proton_cdf = save_data(l3a_output)
            imap_data_access.upload(proton_cdf)

    def process_l3a(self, dependencies: GlowsL3ADependencies) -> GlowsL3LightCurve:
        data = dependencies.data
        l3_data = L3aData(dependencies.ancillary_files)
        l3_data.process_l2_data_file(data)
        l3_data.generate_l3a_data(dependencies.ancillary_files)
        return create_glows_l3a_from_dictionary(l3_data.data, self.input_metadata.to_upstream_data_dependency(
            self.dependencies[0].descriptor))
