import imap_data_access
import numpy as np

from imap_processing.constants import UNSIGNED_INT1_FILL_VALUE, UNSIGNED_INT2_FILL_VALUE
from imap_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_processing.hit.l3.models import HitDirectEventDataProduct
from imap_processing.hit.l3.pha.hit_l3_pha_dependencies import HitL3PhaDependencies
from imap_processing.hit.l3.pha.pha_event_reader import PHAEventReader, RawPHAEvent
from imap_processing.hit.l3.pha.science.calculate_pha import process_pha_event
from imap_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct
from imap_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, rebin_by_pitch_angle_and_gyrophase
from imap_processing.hit.l3.utils import convert_bin_high_low_to_center_delta
from imap_processing.pitch_angles import calculate_unit_vector, calculate_pitch_angle, calculate_gyrophase
from imap_processing.processor import Processor
from imap_processing.utils import save_data


class HitProcessor(Processor):
    def process(self):
        if self.input_metadata.descriptor == "pitch-angle":
            dependencies = HITL3SectoredDependencies.fetch_dependencies(self.dependencies)
            pitch_angle_data_product = self.process_pitch_angle_product(dependencies)
            cdf_file_path = save_data(pitch_angle_data_product)
            imap_data_access.upload(cdf_file_path)
        elif self.input_metadata.descriptor == "direct-event":
            direct_event_dependencies = HitL3PhaDependencies.fetch_dependencies(self.dependencies)
            direct_event_data_product = self.process_direct_event_product(direct_event_dependencies)
            cdf_file_path = save_data(direct_event_data_product)
            imap_data_access.upload(cdf_file_path)
        else:
            raise ValueError(
                f"Don't know how to generate '{self.input_metadata.descriptor}' /n Known HIT l3 data products: 'pitch-angle', 'direct-event'.")

    def process_direct_event_product(self,
                                     direct_event_dependencies: HitL3PhaDependencies) -> HitDirectEventDataProduct:

        epochs = []
        raw_pha_events: list[RawPHAEvent] = []
        for epoch, event_binary in zip(direct_event_dependencies.hit_l1_data.epoch,
                                       direct_event_dependencies.hit_l1_data.event_binary):
            event_raw_pha_events = PHAEventReader.read_all_pha_events(event_binary)
            epochs += [epoch] * len(event_raw_pha_events)
            raw_pha_events += event_raw_pha_events

        charge = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        energy = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        e_delta = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        e_prime = np.full(shape=(len(raw_pha_events)), fill_value=np.nan)
        detected_range = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT1_FILL_VALUE)
        particle_id = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        priority_buffer_number = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT1_FILL_VALUE)
        latency = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT1_FILL_VALUE)
        stim_tag = np.full(shape=(len(raw_pha_events)), fill_value=False)
        long_event_flag = np.full(shape=(len(raw_pha_events)), fill_value=False)
        haz_tag = np.full(shape=(len(raw_pha_events)), fill_value=False)
        a_b_side = np.full(shape=(len(raw_pha_events)), fill_value=False)
        has_unread_adcs = np.full(shape=(len(raw_pha_events)), fill_value=False)
        culling_flag = np.full(shape=(len(raw_pha_events)), fill_value=False)

        pha_value = np.full(shape=(len(raw_pha_events), 64), fill_value=UNSIGNED_INT2_FILL_VALUE)
        energy_at_detector = np.full(shape=(len(raw_pha_events), 64), fill_value=np.nan)
        is_low_gain = np.full(shape=(len(raw_pha_events), 64), fill_value=False)

        detector_flags = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        deindex = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        epindex = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)
        stim_gain = np.full(shape=(len(raw_pha_events)), fill_value=False)
        a_l_stim = np.full(shape=(len(raw_pha_events)), fill_value=False)
        stim_step = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT1_FILL_VALUE)
        dac_value = np.full(shape=(len(raw_pha_events)), fill_value=UNSIGNED_INT2_FILL_VALUE)

        for i, raw_event in enumerate(raw_pha_events):
            event_output = process_pha_event(
                raw_event,
                direct_event_dependencies.cosine_correction_lookup,
                direct_event_dependencies.gain_lookup,
                direct_event_dependencies.range_fit_lookup
            )

            charge[i] = event_output.charge
            energy[i] = event_output.total_energy

            if event_output.e_delta is not None:
                e_delta[i] = event_output.e_delta
            if event_output.e_prime is not None:
                e_prime[i] = event_output.e_prime
            if event_output.detected_range is not None:
                detected_range[i] = event_output.detected_range.value

            particle_id[i] = raw_event.particle_id
            priority_buffer_number[i] = raw_event.priority_buffer_num
            latency[i] = raw_event.time_tag
            stim_tag[i] = raw_event.stim_tag
            long_event_flag[i] = raw_event.long_event_flag
            haz_tag[i] = raw_event.haz_tag
            a_b_side[i] = raw_event.a_b_side_flag
            has_unread_adcs[i] = raw_event.has_unread_adcs
            culling_flag[i] = raw_event.culling_flag

            for event_energy_at_detector, word in zip(event_output.energies, raw_event.pha_words):
                pha_value[i, word.detector.address] = word.adc_value
                energy_at_detector[i, word.detector.address] = event_energy_at_detector
                is_low_gain[i, word.detector.address] = word.is_low_gain

            if raw_event.extended_header is not None:
                detector_flags[i] = raw_event.extended_header.detector_flags
                deindex[i] = raw_event.extended_header.delta_e_index
                epindex[i] = raw_event.extended_header.e_prime_index
            if raw_event.stim_block is not None:
                stim_gain[i] = raw_event.stim_block.stim_gain
                a_l_stim[i] = raw_event.stim_block.a_l_stim
                stim_step[i] = raw_event.stim_block.stim_step
            if raw_event.extended_stim_header is not None:
                dac_value[i] = raw_event.extended_stim_header.dac_value

        return HitDirectEventDataProduct(epoch=epochs,
                                         charge=charge,
                                         energy=energy,
                                         e_delta=e_delta,
                                         e_prime=e_prime,
                                         detected_range=detected_range,
                                         particle_id=particle_id,
                                         priority_buffer_number=priority_buffer_number,
                                         latency=latency,
                                         stim_tag=stim_tag,
                                         long_event_flag=long_event_flag,
                                         haz_tag=haz_tag,
                                         a_b_side=a_b_side,
                                         has_unread_adcs=has_unread_adcs,
                                         culling_flag=culling_flag,
                                         pha_value=pha_value,
                                         energy_at_detector=energy_at_detector,
                                         is_low_gain=is_low_gain,
                                         detector_flags=detector_flags,
                                         deindex=deindex,
                                         epindex=epindex,
                                         stim_gain=stim_gain,
                                         a_l_stim=a_l_stim,
                                         stim_step=stim_step,
                                         dac_value=dac_value,
                                         input_metadata=self.input_metadata.to_upstream_data_dependency("direct-event"))

    def process_pitch_angle_product(self, dependencies: HITL3SectoredDependencies) -> HitPitchAngleDataProduct:
        number_of_pitch_angle_bins = 8
        number_of_gyrophase_bins = 15

        mag_data = dependencies.mag_l1d_data
        hit_data = dependencies.data

        input_flux_data_by_species = {"cno": hit_data.CNO, "helium4": hit_data.helium4, "hydrogen": hit_data.hydrogen,
                                      "iron": hit_data.iron, "NeMgSi": hit_data.NeMgSi}

        rebinned_pa_gyrophase_flux_by_species = {"cno": np.full(hit_data.CNO.shape, np.nan),
                                                 "helium4": np.full(hit_data.helium4.shape, np.nan),
                                                 "hydrogen": np.full(hit_data.hydrogen.shape, np.nan),
                                                 "iron": np.full(hit_data.iron.shape, np.nan),
                                                 "NeMgSi": np.full(hit_data.NeMgSi.shape, np.nan)}

        rebinned_pa_only_flux_by_species = {"cno": np.full(hit_data.CNO.shape[:-1], np.nan),
                                            "helium4": np.full(hit_data.helium4.shape[:-1], np.nan),
                                            "hydrogen": np.full(hit_data.hydrogen.shape[:-1], np.nan),
                                            "iron": np.full(hit_data.iron.shape[:-1], np.nan),
                                            "NeMgSi": np.full(hit_data.NeMgSi.shape[:-1], np.nan)}

        h_energy_center, h_energy_delta = convert_bin_high_low_to_center_delta(hit_data.h_energy_high,
                                                                               hit_data.h_energy_low)
        he4_energy_center, he4_energy_delta = convert_bin_high_low_to_center_delta(hit_data.he4_energy_high,
                                                                                   hit_data.he4_energy_low)
        cno_energy_center, cno_energy_delta = convert_bin_high_low_to_center_delta(hit_data.cno_energy_high,
                                                                                   hit_data.cno_energy_low)
        nemgsi_energy_center, nemgsi_energy_delta = convert_bin_high_low_to_center_delta(hit_data.nemgsi_energy_high,
                                                                                         hit_data.nemgsi_energy_low)
        fe_energy_center, fe_energy_delta = convert_bin_high_low_to_center_delta(hit_data.fe_energy_high,
                                                                                 hit_data.fe_energy_low)

        dec, inc, dec_delta, inc_delta = get_hit_bin_polar_coordinates()
        sector_unit_vectors = get_sector_unit_vectors(dec, inc)
        particle_unit_vectors = -sector_unit_vectors

        pitch_angles, gyrophases, pitch_angle_deltas, gyrophase_delta = get_hit_bin_polar_coordinates(
            number_of_pitch_angle_bins, number_of_gyrophase_bins)

        averaged_mag_data = mag_data.rebin_to(hit_data.epoch, hit_data.epoch_delta)
        for time_index, average_mag_vector in enumerate(averaged_mag_data):
            mag_unit_vector = calculate_unit_vector(average_mag_vector)
            input_bin_pitch_angles = calculate_pitch_angle(particle_unit_vectors, mag_unit_vector)
            input_bin_gyrophases = calculate_gyrophase(particle_unit_vectors, mag_unit_vector)

            for species, flux in input_flux_data_by_species.items():
                pa_gyrophase, pa_only = rebin_by_pitch_angle_and_gyrophase(flux[time_index], input_bin_pitch_angles,
                                                                           input_bin_gyrophases,
                                                                           number_of_pitch_angle_bins,
                                                                           number_of_gyrophase_bins)
                rebinned_pa_gyrophase_flux_by_species[species][time_index] = pa_gyrophase
                rebinned_pa_only_flux_by_species[species][time_index] = pa_only

        return HitPitchAngleDataProduct(self.input_metadata.to_upstream_data_dependency("sci"), hit_data.epoch,
                                        hit_data.epoch_delta, pitch_angles, pitch_angle_deltas,
                                        gyrophases,
                                        gyrophase_delta,
                                        rebinned_pa_gyrophase_flux_by_species["hydrogen"],
                                        rebinned_pa_only_flux_by_species["hydrogen"],
                                        h_energy_center,
                                        h_energy_delta,
                                        rebinned_pa_gyrophase_flux_by_species["helium4"],
                                        rebinned_pa_only_flux_by_species["helium4"],
                                        he4_energy_center, he4_energy_delta,
                                        rebinned_pa_gyrophase_flux_by_species["cno"],
                                        rebinned_pa_only_flux_by_species["cno"],
                                        cno_energy_center, cno_energy_delta,
                                        rebinned_pa_gyrophase_flux_by_species["NeMgSi"],
                                        rebinned_pa_only_flux_by_species["NeMgSi"], nemgsi_energy_center,
                                        nemgsi_energy_delta, rebinned_pa_gyrophase_flux_by_species["iron"],
                                        rebinned_pa_only_flux_by_species["iron"], fe_energy_center,
                                        fe_energy_delta)
