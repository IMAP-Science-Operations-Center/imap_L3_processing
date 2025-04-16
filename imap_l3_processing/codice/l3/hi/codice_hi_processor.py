import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3_dependencies import CodiceHiL3Dependencies
from imap_l3_processing.codice.l3.hi.models import CodiceL3HiDirectEvents, CodiceL3HiDirectEventsBuilder
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.pitch_angles import calculate_unit_vector
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class CodiceHiProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            dependencies = CodiceHiL3Dependencies.fetch_dependencies(self.dependencies)
            processed_codice_direct_events = self.process_l3a(dependencies)
            saved_cdf = save_data(processed_codice_direct_events)
            upload(saved_cdf)

    def process_l3a(self, dependencies: CodiceHiL3Dependencies) -> CodiceL3HiDirectEvents:
        tof_lookup = dependencies.tof_lookup
        l2_data = dependencies.codice_l2_hi_data

        estimated_mass_with_bounds = []
        energy_per_nucleons_per_priority_event = []
        for index, priority_event in enumerate(l2_data.priority_events):
            tof = priority_event.time_of_flight
            energy_per_nuc_with_bounds = np.array([e for t in tof.flat for e in tof_lookup[t]]).reshape((*tof.shape, 3))
            estimated_mass_with_bounds.append(priority_event.ssd_energy[:, :, np.newaxis] / energy_per_nuc_with_bounds)
            energy_per_nucleons_per_priority_event.append(energy_per_nuc_with_bounds)

        # @formatter:off
        return (CodiceL3HiDirectEventsBuilder(l2_data)
                            .updated_priority_event_0(energy_per_nucleons_per_priority_event[0], estimated_mass_with_bounds[0])
                            .updated_priority_event_1(energy_per_nucleons_per_priority_event[1], estimated_mass_with_bounds[1])
                            .updated_priority_event_2(energy_per_nucleons_per_priority_event[2], estimated_mass_with_bounds[2])
                            .updated_priority_event_3(energy_per_nucleons_per_priority_event[3], estimated_mass_with_bounds[3])
                            .updated_priority_event_4(energy_per_nucleons_per_priority_event[4], estimated_mass_with_bounds[4])
                            .updated_priority_event_5(energy_per_nucleons_per_priority_event[5], estimated_mass_with_bounds[5])
                            .convert())

    def process_l3b(self, dependencies: CodicePitchAngleDependencies):
        mag_data = dependencies.mag_l1d_data
        sectored_intensities = dependencies.codice_sectored_intensities_data
        rebinned_mag_data = mag_data.rebin_to(sectored_intensities.epoch, sectored_intensities.epoch_delta)

        mag_unit_vectors = calculate_unit_vector(rebinned_mag_data)

        return None
