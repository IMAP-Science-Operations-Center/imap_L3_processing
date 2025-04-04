import numpy as np

from imap_l3_processing.codice.l2.direct_event.codice_l2_dependencies import CodiceL2Dependencies
from imap_l3_processing.codice.models import CodiceL2HiDataProduct
from imap_l3_processing.models import UpstreamDataDependency, InputMetadata
from imap_l3_processing.processor import Processor


class CodiceProcessor(Processor):
    def __init__(self, dependencies: list[UpstreamDataDependency], input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        pass
        # fetch dependencies
        # call process_l2
        # save_cdf(DATA_PRODUCT)

    def process_l2(self, codice_l2_dependencies: CodiceL2Dependencies):
        codice_l1a_hi_data = codice_l2_dependencies.codice_l1a_hi_data
        energy_look_up = codice_l2_dependencies.energy_lookup_table

        priority_events = [codice_l1a_hi_data.priority_event_0, codice_l1a_hi_data.priority_event_1,
                           codice_l1a_hi_data.priority_event_2, codice_l1a_hi_data.priority_event_3,
                           codice_l1a_hi_data.priority_event_4, codice_l1a_hi_data.priority_event_5]

        epochs = codice_l1a_hi_data.epochs
        energy_in_mev = np.full((len(epochs), len(priority_events), len(priority_events[0].ssd_energy)), np.nan)
        for epoch_index, epoch in enumerate(epochs):
            for priority_index, priority_event in enumerate(priority_events):
                ssd_id = priority_event.ssd_id[epoch_index]
                ssd_energy = priority_event.ssd_energy[epoch_index]
                energy_range = priority_event.energy_range[epoch_index]
                for i in range(len(ssd_id)):
                    converted_energy = codice_l2_dependencies.energy_lookup_table.convert_to_mev(ssd_id[i],
                                                                                                 energy_range[i],
                                                                                                 ssd_energy[i])
                    energy_in_mev[epoch_index][priority_index][i] = converted_energy

        return CodiceL2HiDataProduct(input_metadata=None, epoch=None, data_quality=None, energy_range=None,
                                     multi_flag=None,
                                     number_of_events=None, energy=energy_in_mev, spin_sector=None, spin_number=None,
                                     time_of_flight=None, priority=None)
