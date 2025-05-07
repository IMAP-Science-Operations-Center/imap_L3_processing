from collections import namedtuple

import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2DirectEventData, \
    CodiceLoL3aDirectEventDataProduct, CodiceLoPartialDensityData, CodiceLoL3aRatiosDataProduct
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_normalization_ratio, calculate_total_number_of_events, calculate_mass, calculate_mass_per_charge
from imap_l3_processing.hi.l3.models import safe_divide
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data

PriorityRate = namedtuple('PriorityRate', ('epoch', 'energy_table', 'priority_count'))


class CodiceLoProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.descriptor == "lo-partial-densities":
            dependencies = CodiceLoL3aPartialDensitiesDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_partial_densities(dependencies)
        elif self.input_metadata.descriptor == "lo-direct-events":
            dependencies = CodiceLoL3aDirectEventsDependencies.fetch_dependencies(self.dependencies)
            data_product = self._process_l3a_direct_event_data_product(dependencies)
        elif self.input_metadata.descriptor == "lo-sw-ratios":
            dependencies = CodiceLoL3aRatiosDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_ratios(dependencies)
        else:
            raise NotImplementedError

        data_product.parent_file_names = self.get_parent_file_names()
        saved_l3a_cdf = save_data(data_product)
        upload(saved_l3a_cdf)

    def process_l3a_ratios(self, dependencies: CodiceLoL3aRatiosDependencies) -> CodiceLoL3aRatiosDataProduct:
        input_data = dependencies.partial_density_data
        oxygen_density = input_data.oplus5_partial_density + input_data.oplus6_partial_density + input_data.oplus7_partial_density + input_data.oplus8_partial_density
        iron_density = input_data.fe_hiq_partial_density + input_data.fe_loq_partial_density
        carbon_density = input_data.cplus4_partial_density + input_data.cplus5_partial_density + input_data.cplus6_partial_density
        return CodiceLoL3aRatiosDataProduct(
            input_metadata=self.input_metadata,
            epoch=input_data.epoch,
            epoch_delta=input_data.epoch_delta,
            c_to_o_ratio=carbon_density / oxygen_density,
            mg_to_o_ratio=input_data.mg_partial_density / oxygen_density,
            fe_to_o_ratio=iron_density / oxygen_density,
            c6_to_c5_ratio=input_data.cplus6_partial_density / input_data.cplus5_partial_density,
            c6_to_c4_ratio=input_data.cplus6_partial_density / input_data.cplus4_partial_density,
            o7_to_o6_ratio=input_data.oplus7_partial_density / input_data.oplus6_partial_density,
            felo_to_fehi_ratio=input_data.fe_loq_partial_density / input_data.fe_hiq_partial_density,
        )

    def process_l3a_partial_densities(self, dependencies: CodiceLoL3aPartialDensitiesDependencies):
        codice_lo_l2_data = dependencies.codice_l2_lo_data
        mass_per_charge_lookup = dependencies.mass_per_charge_lookup
        h_plus_partial_density = calculate_partial_densities(codice_lo_l2_data.hplus, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.hplus)
        heplusplus_partial_density = calculate_partial_densities(codice_lo_l2_data.heplusplus,
                                                                 codice_lo_l2_data.energy_table,
                                                                 mass_per_charge_lookup.heplusplus)
        cplus4_partial_density = calculate_partial_densities(codice_lo_l2_data.cplus4, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.cplus4)
        cplus5_partial_density = calculate_partial_densities(codice_lo_l2_data.cplus5, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.cplus5)
        cplus6_partial_density = calculate_partial_densities(codice_lo_l2_data.cplus6, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.cplus6)
        oplus5_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus5, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.oplus5)
        oplus6_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus6, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.oplus6)
        oplus7_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus7, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.oplus7)
        oplus8_partial_density = calculate_partial_densities(codice_lo_l2_data.oplus8, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.oplus8)
        ne_partial_density = calculate_partial_densities(codice_lo_l2_data.ne, codice_lo_l2_data.energy_table,
                                                         mass_per_charge_lookup.ne)
        mg_partial_density = calculate_partial_densities(codice_lo_l2_data.mg, codice_lo_l2_data.energy_table,
                                                         mass_per_charge_lookup.mg)
        si_partial_density = calculate_partial_densities(codice_lo_l2_data.si, codice_lo_l2_data.energy_table,
                                                         mass_per_charge_lookup.si)
        fe_loq_partial_density = calculate_partial_densities(codice_lo_l2_data.fe_loq, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.fe_loq)
        fe_hiq_partial_density = calculate_partial_densities(codice_lo_l2_data.fe_hiq, codice_lo_l2_data.energy_table,
                                                             mass_per_charge_lookup.fe_hiq)

        return CodiceLoL3aPartialDensityDataProduct(
            input_metadata=self.input_metadata,
            data=CodiceLoPartialDensityData(
                epoch=codice_lo_l2_data.epoch,
                epoch_delta=codice_lo_l2_data.epoch_delta_plus,
                hplus_partial_density=h_plus_partial_density,
                heplusplus_partial_density=heplusplus_partial_density,
                cplus4_partial_density=cplus4_partial_density,
                cplus5_partial_density=cplus5_partial_density,
                cplus6_partial_density=cplus6_partial_density,
                oplus5_partial_density=oplus5_partial_density,
                oplus6_partial_density=oplus6_partial_density,
                oplus7_partial_density=oplus7_partial_density,
                oplus8_partial_density=oplus8_partial_density,
                ne_partial_density=ne_partial_density,
                mg_partial_density=mg_partial_density,
                si_partial_density=si_partial_density,
                fe_loq_partial_density=fe_loq_partial_density,
                fe_hiq_partial_density=fe_hiq_partial_density,
            )
        )

    def _process_l3a_direct_event_data_product(self,
                                               dependencies: CodiceLoL3aDirectEventsDependencies) -> CodiceLoL3aDirectEventDataProduct:
        codice_sw_priority_rates_l1a_data = dependencies.codice_lo_l1a_sw_priority_rates
        codice_nsw_priority_rates_l1a_data = dependencies.codice_lo_l1a_nsw_priority_rates
        codice_direct_events: CodiceLoL2DirectEventData = dependencies.codice_l2_direct_events
        event_buffer = codice_direct_events.priority_events[0].tof.shape[-1]
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
            np.full((len(codice_direct_events.epoch), len(priority_rates_for_events), event_buffer), np.nan)
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
