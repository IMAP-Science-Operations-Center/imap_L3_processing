from collections import namedtuple

import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2DirectEventData, \
    CodiceLoL3aDirectEventDataProduct, CodiceLoPartialDensityData, CodiceLoL3aRatiosDataProduct, \
    CodiceLoL3ChargeStateDistributionsDataProduct
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_normalization_ratio, calculate_total_number_of_events, calculate_mass, calculate_mass_per_charge
from imap_l3_processing.data_utils import safe_divide
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
            data_product = self.process_l3a_direct_event_data_product(dependencies)
        elif self.input_metadata.descriptor == "lo-sw-ratios":
            dependencies = CodiceLoL3aRatiosDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_ratios(dependencies)
        elif self.input_metadata.descriptor == "lo-sw-abundances":
            dependencies = CodiceLoL3aRatiosDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_abundances(dependencies)
        else:
            raise NotImplementedError(
                f"Unknown data level and descriptor for CoDICE: {self.input_metadata.data_level}, {self.input_metadata.descriptor}")

        data_product.parent_file_names = self.get_parent_file_names()
        saved_l3a_cdf = save_data(data_product)
        upload(saved_l3a_cdf)

    def process_l3a_ratios(self, dependencies: CodiceLoL3aRatiosDependencies) -> CodiceLoL3aRatiosDataProduct:
        input_data = dependencies.partial_density_data
        c4 = _average_over_block(input_data.cplus4_partial_density, 3)
        c5 = _average_over_block(input_data.cplus5_partial_density, 3)
        c6 = _average_over_block(input_data.cplus6_partial_density, 3)
        o5 = _average_over_block(input_data.oplus5_partial_density, 3)
        o6 = _average_over_block(input_data.oplus6_partial_density, 3)
        o7 = _average_over_block(input_data.oplus7_partial_density, 3)
        o8 = _average_over_block(input_data.oplus8_partial_density, 3)
        feloq = _average_over_block(input_data.fe_loq_partial_density, 3)
        fehiq = _average_over_block(input_data.fe_hiq_partial_density, 3)
        mg = _average_over_block(input_data.mg_partial_density, 3)

        return CodiceLoL3aRatiosDataProduct(
            input_metadata=self.input_metadata,
            epoch=_average_dates_over_block(input_data.epoch, 3),
            epoch_delta=_sum_over_block(input_data.epoch_delta, 3),
            c_to_o_ratio=safe_divide(c4 + c5 + c6, o5 + o6 + o7 + o8),
            mg_to_o_ratio=safe_divide(mg, o5 + o6 + o7 + o8),
            fe_to_o_ratio=safe_divide(feloq + fehiq, o5 + o6 + o7 + o8),
            c6_to_c5_ratio=safe_divide(c6, c5),
            c6_to_c4_ratio=safe_divide(c6, c4),
            o7_to_o6_ratio=safe_divide(o7, o6),
            felo_to_fehi_ratio=safe_divide(feloq, fehiq),
        )

    def process_l3a_abundances(self,
                               dependencies: CodiceLoL3aRatiosDependencies) -> CodiceLoL3ChargeStateDistributionsDataProduct:

        o5 = dependencies.partial_density_data.oplus5_partial_density
        o6 = dependencies.partial_density_data.oplus6_partial_density
        o7 = dependencies.partial_density_data.oplus7_partial_density
        o8 = dependencies.partial_density_data.oplus8_partial_density
        c4 = dependencies.partial_density_data.cplus4_partial_density
        c5 = dependencies.partial_density_data.cplus5_partial_density
        c6 = dependencies.partial_density_data.cplus6_partial_density

        o_densities = _average_over_block(np.column_stack((o5, o6, o7, o8)), block_size=3)
        o_distribution = safe_divide(o_densities, np.sum(o_densities, axis=1, keepdims=True))

        c_densities = _average_over_block(np.column_stack((c4, c5, c6)), block_size=3)
        c_distribution = safe_divide(c_densities, np.sum(c_densities, axis=1, keepdims=True))

        return CodiceLoL3ChargeStateDistributionsDataProduct(
            self.input_metadata,
            epoch=_average_dates_over_block(dependencies.partial_density_data.epoch, 3),
            epoch_delta=_sum_over_block(dependencies.partial_density_data.epoch_delta, 3),
            oxygen_charge_state_distribution=o_distribution,
            carbon_charge_state_distribution=c_distribution
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

    def process_l3a_direct_event_data_product(self,
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
        try:
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
                priority_event_binned_by_energy_and_spin_angle = priority_event.total_events_binned_by_energy_step_and_spin_angle()
                for e in range(len(codice_direct_events.epoch)):
                    norm = calculate_normalization_ratio(
                        priority_event_binned_by_energy_and_spin_angle[e],
                        total_by_epoch[e])

                    normalization[e][priority_index] = norm
        except Exception as e:
            print(e)

        return CodiceLoL3aDirectEventDataProduct(
            input_metadata=self.input_metadata,
            epoch=codice_direct_events.epoch,
            epoch_delta=codice_direct_events.epoch_delta_plus,
            normalization=normalization,
            mass_per_charge=mass_per_charge,
            mass=mass,
            event_energy=energy,
            gain=gain,
            apd_id=apd_id,
            multi_flag=multi_flag,
            num_events=num_events,
            tof=tof,
            data_quality=data_quality,
        )


def _average_over_block(data_array: np.ndarray, block_size: int):
    return np.array([data_array[i:i + block_size].mean(axis=0) for i in range(0, len(data_array), block_size)])


def _sum_over_block(data_array: np.ndarray, block_size: int):
    return np.array([data_array[i:i + block_size].sum() for i in range(0, len(data_array), block_size)])


def _average_dates_over_block(data_array: np.ndarray, block_size: int):
    dates_as_int = data_array.astype('datetime64[us]').astype('int64')
    return np.array([dates_as_int[i:i + block_size].mean() for i in range(0, len(dates_as_int), block_size)]).astype(
        "datetime64[us]").astype('O')
