from collections import namedtuple

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

PriorityRate = namedtuple('PriorityRate', ('epoch', 'energy_table', 'priority_count'))


class CodiceLoProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        dependencies = CodiceLoL3aDependencies.fetch_dependencies(self.dependencies)
        l3a_data = self.process_l3a(dependencies)
        l3a_data.parent_file_names = self.get_parent_file_names()
        l3a_direct_event_data = self._process_l3a_direct_event_data_product(dependencies)

        saved_l3a_cdf = save_data(l3a_data)
        saved_l3a_direct_event_cdf = save_data(l3a_direct_event_data)

        upload(saved_l3a_cdf)
        upload(saved_l3a_direct_event_cdf)

    def process_l3a(self, dependencies: CodiceLoL3aDependencies):
        codice_lo_l2_data = dependencies.codice_l2_lo_data
        mass_per_charge_lookup = dependencies.mass_per_charge_lookup
        return CodiceLoL3aPartialDensityDataProduct(
            input_metadata=self.input_metadata,
            epoch=codice_lo_l2_data.epoch,
            epoch_delta_plus=codice_lo_l2_data.epoch_delta_plus,
            epoch_delta_minus=codice_lo_l2_data.epoch_delta_minus,
            hplus_partial_density=calculate_partial_densities(
                codice_lo_l2_data.hplus,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.hplus),
            heplusplus_partial_density=calculate_partial_densities(
                codice_lo_l2_data.heplusplus,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.heplusplus),
            cplus4_partial_density=calculate_partial_densities(
                codice_lo_l2_data.cplus4,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.cplus4),
            cplus5_partial_density=calculate_partial_densities(
                codice_lo_l2_data.cplus5,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.cplus5),
            cplus6_partial_density=calculate_partial_densities(
                codice_lo_l2_data.cplus6,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.cplus6),
            oplus5_partial_density=calculate_partial_densities(
                codice_lo_l2_data.oplus5,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.oplus5),
            oplus6_partial_density=calculate_partial_densities(
                codice_lo_l2_data.oplus6,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.oplus6),
            oplus7_partial_density=calculate_partial_densities(
                codice_lo_l2_data.oplus7,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.oplus7),
            oplus8_partial_density=calculate_partial_densities(
                codice_lo_l2_data.oplus8,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.oplus8),
            mg_partial_density=calculate_partial_densities(
                codice_lo_l2_data.mg,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.mg),
            si_partial_density=calculate_partial_densities(
                codice_lo_l2_data.si,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.si),
            fe_loq_partial_density=calculate_partial_densities(
                codice_lo_l2_data.fe_loq,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.fe_loq),
            fe_hiq_partial_density=calculate_partial_densities(
                codice_lo_l2_data.fe_hiq,
                codice_lo_l2_data.energy_table,
                mass_per_charge_lookup.fe_hiq))

    def _process_l3a_direct_event_data_product(self,
                                               dependencies: CodiceLoL3aDependencies) -> CodiceLoL3aDirectEventDataProduct:
        codice_sw_priority_rates_l1a_data = dependencies.codice_lo_l1a_sw_priority_rates
        codice_nsw_priority_rates_l1a_data = dependencies.codice_lo_l1a_nsw_priority_rates
        codice_direct_events: CodiceLoL2DirectEventData = dependencies.codice_l2_direct_events
        event_number = codice_direct_events.event_num
        mass_coefficient_lookup = dependencies.mass_coefficient_lookup
        priority_rates_for_events = [
            codice_sw_priority_rates_l1a_data.p0_tcrs,
            codice_sw_priority_rates_l1a_data.p1_hplus,
            codice_sw_priority_rates_l1a_data.p2_heplusplus,
            codice_sw_priority_rates_l1a_data.p3_heavies,
            codice_sw_priority_rates_l1a_data.p4_dcrs,
            codice_nsw_priority_rates_l1a_data.p5_heavies,
            codice_nsw_priority_rates_l1a_data.p6_hplus_heplusplus
        ]

        normalization = np.full((len(codice_direct_events.epoch), len(priority_rates_for_events), 128, 12), np.nan)

        (mass_per_charge,
         mass,
         energy,
         gain,
         apd_id,
         spin_angle,
         multi_flag,
         pha_type,
         tof) = [
            np.full((len(codice_direct_events.epoch), len(priority_rates_for_events), len(event_number)), np.nan)
            for _ in range(9)]

        (data_quality, num_events) = [np.full((len(codice_direct_events.epoch), len(priority_rates_for_events)), np.nan)
                                      for _ in range(2)]

        for priority_index, (priority_event, priority_rate) in enumerate(
                zip(codice_direct_events.priority_events, priority_rates_for_events)):

            total_by_epoch: np.ndarray[int] = calculate_total_number_of_events(priority_rate,
                                                                               codice_sw_priority_rates_l1a_data.acquisition_time_per_step)

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
