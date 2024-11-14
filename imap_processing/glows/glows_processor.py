import dataclasses

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
            data = read_l2_glows_data(l3a_dependencies.data)
            light_curve = self.process_l3a(data, l3a_dependencies)
            proton_cdf = save_data(light_curve)