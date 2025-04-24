import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2DirectEventData, \
    CodiceLoL3aDirectEventDataProduct
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_normalization_ratio, calculate_total_number_of_events, calculate_mass, calculate_mass_per_charge
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class CodiceLoProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        dependencies = CodiceLoL3aDependencies.fetch_dependencies(self.dependencies)
        l3a_data = self.process_l3a(dependencies)
        l3a_direct_event_data = self._process_l3a_direct_event_data_product(dependencies)

        saved_l3a_cdf = save_data(l3a_data)
        saved_l3a_direct_event_cdf = save_data(l3a_direct_event_data)

        upload(saved_l3a_cdf)
        upload(saved_l3a_direct_event_cdf)

    def process_l3a(self, dependencies: CodiceLoL3aDependencies):

        for species_name, species_intensities in dependencies.codice_l2_lo_data.get_species_intensities().items():
            partial_density = calculate_partial_densities(species_intensities)

            match species_name:
                case "H+":
                    h_partial_density = partial_density
                case "He++":
                    he_partial_density = partial_density
                case "C+4":
                    c4_partial_density = partial_density
                case "C+5":
                    c5_partial_density = partial_density
                case "C+6":
                    c6_partial_density = partial_density
                case "O+5":
                    o5_partial_density = partial_density
                case "O+6":
                    o6_partial_density = partial_density
                case "O+7":
                    o7_partial_density = partial_density
                case "O+8":
                    o8_partial_density = partial_density
                case "Mg":
                    mg_partial_density = partial_density
                case "Si":
                    si_partial_density = partial_density
                case "Fe (low Q)":
                    fe_low_partial_density = partial_density
                case "Fe (high Q)":
                    fe_high_partial_density = partial_density
                case _:
                    raise NotImplementedError
        epoch = np.array([np.nan])
        epoch_delta = np.full(len(epoch), 4.8e+11)
        return CodiceLoL3aPartialDensityDataProduct(epoch=epoch, epoch_delta=epoch_delta,
                                                    h_partial_density=h_partial_density,
                                                    he_partial_density=he_partial_density,
                                                    c4_partial_density=c4_partial_density,
                                                    c5_partial_density=c5_partial_density,
                                                    c6_partial_density=c6_partial_density,
                                                    o5_partial_density=o5_partial_density,
                                                    o6_partial_density=o6_partial_density,
                                                    o7_partial_density=o7_partial_density,
                                                    o8_partial_density=o8_partial_density,
                                                    mg_partial_density=mg_partial_density,
                                                    si_partial_density=si_partial_density,
                                                    fe_low_partial_density=fe_low_partial_density,
                                                    fe_high_partial_density=fe_high_partial_density)

    def _process_l3a_direct_event_data_product(self,
                                               dependencies: CodiceLoL3aDependencies) -> CodiceLoL3aDirectEventDataProduct:
        codice_priority_rates_l2_data = dependencies.codice_l2b_lo_priority_rates
        codice_direct_events: CodiceLoL2DirectEventData = dependencies.codice_l2_direct_events
        event_number = codice_direct_events.event_num
        mass_coefficient_lookup = dependencies.mass_coefficient_lookup
        priority_rates_for_events = [
            codice_priority_rates_l2_data.lo_sw_priority_p0_tcrs,
            codice_priority_rates_l2_data.lo_sw_priority_p1_hplus,
            codice_priority_rates_l2_data.lo_sw_priority_p2_heplusplus,
            codice_priority_rates_l2_data.lo_sw_priority_p3_heavies,
            codice_priority_rates_l2_data.lo_sw_priority_p4_dcrs,
            codice_priority_rates_l2_data.lo_nsw_priority_p5_heavies,
            codice_priority_rates_l2_data.lo_nsw_priority_p6_hplus_heplusplus,
            codice_priority_rates_l2_data.lo_nsw_priority_p7_missing,
        ]

        normalization = np.full((len(codice_direct_events.epoch), len(priority_rates_for_events), 128, 12), np.nan)

        (mass_per_charge, mass, energy, gain, apd_id, multi_flag, pha_type,
         tof) = [
            np.full((len(codice_direct_events.epoch), len(priority_rates_for_events), len(event_number)),
                    np.nan) for _ in range(8)]

        (data_quality, num_events) = [np.full((len(codice_direct_events.epoch), len(priority_rates_for_events)), np.nan)
                                      for _ in range(2)]

        for priority_index, (priority_event, priority_rate) in enumerate(
                zip(codice_direct_events.priority_events, priority_rates_for_events)):

            total_by_epoch: np.ndarray[int] = calculate_total_number_of_events(priority_rate,
                                                                               codice_priority_rates_l2_data.acquisition_times)

            mass_per_charge[:, priority_index, :] = calculate_mass_per_charge(priority_event)
            mass[:, priority_index, :] = calculate_mass(priority_event, mass_coefficient_lookup)
            energy[:, priority_index, :] = priority_event.apd_energy
            gain[:, priority_index, :] = priority_event.apd_gain
            apd_id[:, priority_index, :] = priority_event.apd_id
            multi_flag[:, priority_index, :] = priority_event.multi_flag
            pha_type[:, priority_index, :] = priority_event.pha_type
            tof[:, priority_index, :] = priority_event.tof

            data_quality[:, priority_index] = priority_event.data_quality
            num_events[:, priority_index] = priority_event.num_events

            for e in range(len(codice_direct_events.epoch)):
                norm = calculate_normalization_ratio(
                    priority_event.total_events_binned_by_energy_step_and_spin_angle()[e],
                    total_by_epoch[e])

                normalization[e][priority_index] = norm

        return CodiceLoL3aDirectEventDataProduct(
            input_metadata=self.input_metadata,
            epoch=codice_direct_events.epoch,
            event_num=codice_direct_events.event_num,
            normalization=normalization,
            mass_per_charge=mass_per_charge,
            mass=mass,
            energy=energy,
            gain=gain,
            apd_id=apd_id,
            multi_flag=multi_flag,
            num_events=num_events,
            pha_type=pha_type,
            tof=tof,
            data_quality=data_quality,
        )
