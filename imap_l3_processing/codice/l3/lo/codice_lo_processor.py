from collections import namedtuple

import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies import \
    CodiceLoL3a3dDistributionsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies
from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup, \
    PositionToElevationLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.efficiency_lookup import EfficiencyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2DirectEventData, \
    CodiceLoL3aDirectEventDataProduct, CodiceLoPartialDensityData, CodiceLoL3aRatiosDataProduct, \
    CodiceLoL3ChargeStateDistributionsDataProduct, CODICE_LO_L2_NUM_PRIORITIES, CodiceLoL3a3dDistributionDataProduct
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_mass, calculate_mass_per_charge, \
    rebin_counts_by_energy_and_spin_angle, rebin_to_counts_by_species_elevation_and_spin_sector, normalize_counts, \
    combine_priorities_and_convert_to_rate, rebin_3d_distribution_azimuth_to_elevation, convert_count_rate_to_intensity, \
    CODICE_LO_NUM_AZIMUTH_BINS
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
        elif self.input_metadata.descriptor == "lo-3d-instrument-frame":
            dependencies = CodiceLoL3a3dDistributionsDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_3d_distribution_product(dependencies)
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

        (mass_per_charge,
         mass,
         energy,
         gain,
         apd_id,
         spin_angle,
         elevation,
         multi_flag,
         pha_type,
         tof) = [
            np.full((len(codice_direct_events.epoch), len(priority_rates_for_events), event_buffer), np.nan)
            for _ in range(10)]

        (data_quality, num_events) = [np.full((len(codice_direct_events.epoch), len(priority_rates_for_events)), np.nan)
                                      for _ in range(2)]

        energy_lut = EnergyLookup.from_bin_centers(codice_sw_priority_rates_l1a_data.energy_table)
        spin_angle_lut = SpinAngleLookup()
        normalization = np.full((len(codice_direct_events.epoch), CODICE_LO_L2_NUM_PRIORITIES,
                                 energy_lut.num_bins, spin_angle_lut.num_bins), np.nan)

        try:

            for priority_index, (priority_event, priority_counts_total_count) in enumerate(
                    zip(codice_direct_events.priority_events, priority_rates_for_events)):
                mass_per_charge[:, priority_index, :] = calculate_mass_per_charge(priority_event)
                mass[:, priority_index, :] = calculate_mass(priority_event, mass_coefficient_lookup)
                energy[:, priority_index, :] = priority_event.apd_energy
                gain[:, priority_index, :] = priority_event.apd_gain
                apd_id[:, priority_index, :] = priority_event.apd_id
                multi_flag[:, priority_index, :] = priority_event.multi_flag
                tof[:, priority_index, :] = priority_event.tof
                spin_angle[:, priority_index, :] = priority_event.spin_angle
                elevation[:, priority_index, :] = priority_event.elevation
                data_quality[:, priority_index] = priority_event.data_quality
                num_events[:, priority_index] = priority_event.num_events

                direct_events_binned_by_energy_and_spin = rebin_counts_by_energy_and_spin_angle(priority_event,
                                                                                                spin_angle_lut,
                                                                                                energy_lut)
                normalization[:, priority_index, ...] = \
                    priority_counts_total_count / direct_events_binned_by_energy_and_spin

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
            spin_angle=spin_angle,
            elevation=elevation
        )

    def process_l3a_3d_distribution_product(self, dependencies: CodiceLoL3a3dDistributionsDependencies):

        l3a_de_mass = dependencies.l3a_direct_event_data.mass
        l3a_de_mass_per_charge = dependencies.l3a_direct_event_data.mass_per_charge
        l3a_de_energy = dependencies.l3a_direct_event_data.event_energy
        l3a_de_spin_angle = dependencies.l3a_direct_event_data.spin_angle
        l3a_de_apd_id = dependencies.l3a_direct_event_data.apd_id
        l3a_de_normalization = dependencies.l3a_direct_event_data.normalization
        l3a_de_num_events = dependencies.l3a_direct_event_data.num_events

        l1a_energy_table = dependencies.l1a_sw_data.energy_table
        l1a_acquisition_time = dependencies.l1a_sw_data.acquisition_time_per_step

        mass_species_bin_lookup = dependencies.mass_species_bin_lookup
        spin_angle_lut = SpinAngleLookup()
        position_elevation_lut = PositionToElevationLookup()
        energy_lut = EnergyLookup.from_bin_centers(l1a_energy_table)
        geometric_factor_lut = dependencies.geometric_factors_lookup

        counts_3d_data = rebin_to_counts_by_species_elevation_and_spin_sector(
            mass=l3a_de_mass,
            mass_per_charge=l3a_de_mass_per_charge,
            energy=l3a_de_energy,
            spin_angle=l3a_de_spin_angle,
            apd_id=l3a_de_apd_id,
            mass_species_bin_lookup=mass_species_bin_lookup,
            spin_angle_lut=spin_angle_lut,
            position_elevation_lut=position_elevation_lut,
            energy_lut=energy_lut,
            num_events=l3a_de_num_events,
        )

        normalized_counts = normalize_counts(counts_3d_data, l3a_de_normalization)
        normalized_count_rates = combine_priorities_and_convert_to_rate(normalized_counts, l1a_acquisition_time)

        efficiency_lookup = EfficiencyLookup.create_with_fake_data(mass_species_bin_lookup.get_num_species(),
                                                                   CODICE_LO_NUM_AZIMUTH_BINS,
                                                                   energy_lut.num_bins)

        l1_sw_rgfo_half_spins = dependencies.l1a_sw_data.rgfo_half_spin
        geometric_factors = geometric_factor_lut.get_geometric_factors(l1_sw_rgfo_half_spins)

        intensities = convert_count_rate_to_intensity(normalized_count_rates, efficiency_lookup, geometric_factors)
        rebin_3d_distribution_azimuth_to_elevation(intensities, position_elevation_lut)

        return CodiceLoL3a3dDistributionDataProduct(
            input_metadata=self.input_metadata,
            epoch=dependencies.l3a_direct_event_data.epoch,
            epoch_delta=dependencies.l3a_direct_event_data.epoch_delta,
            elevation=position_elevation_lut.bin_centers,
            elevation_delta=position_elevation_lut.bin_deltas,
            spin_angle=spin_angle_lut.bin_centers,
            spin_angle_delta=spin_angle_lut.bin_deltas,
            energy=energy_lut.bin_centers,
            energy_delta_plus=energy_lut.delta_plus,
            energy_delta_minus=energy_lut.delta_minus
        )


def _average_over_block(data_array: np.ndarray, block_size: int):
    return np.array([data_array[i:i + block_size].mean(axis=0) for i in range(0, len(data_array), block_size)])


def _sum_over_block(data_array: np.ndarray, block_size: int):
    return np.array([data_array[i:i + block_size].sum() for i in range(0, len(data_array), block_size)])


def _average_dates_over_block(data_array: np.ndarray, block_size: int):
    dates_as_int = data_array.astype('datetime64[us]').astype('int64')
    return np.array([dates_as_int[i:i + block_size].mean() for i in range(0, len(dates_as_int), block_size)]).astype(
        "datetime64[us]").astype('O')
