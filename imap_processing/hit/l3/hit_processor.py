import imap_data_access
import numpy as np

from imap_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_processing.hit.l3.models import HitDirectEventDataProduct
from imap_processing.hit.l3.pha.hit_l3_pha_dependencies import HitL3PhaDependencies
from imap_processing.hit.l3.pha.pha_event_reader import PHAEventReader
from imap_processing.hit.l3.pha.science.calculate_pha import process_pha_event
from imap_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct
from imap_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, calculate_sector_areas, rebin_by_pitch_angle_and_gyrophase
from imap_processing.hit.l3.utils import convert_bin_high_low_to_center_delta
from imap_processing.pitch_angles import calculate_unit_vector, calculate_pitch_angle, calculate_gyrophase
from imap_processing.processor import Processor
from imap_processing.utils import save_data


class HitProcessor(Processor):
    def process(self):
        if self.input_metadata.descriptor == "pitch-angle":
            pitch_angle_data_product = self.process_pitch_angle_product()
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
        processed_events = []
        for event_binary in direct_event_dependencies.hit_l1_data.event_binary:
            raw_pha_events = PHAEventReader.read_all_pha_events(event_binary)
            for raw_event in raw_pha_events:
                processed_events.append(process_pha_event(raw_event, direct_event_dependencies.cosine_correction_lookup,
                                                          direct_event_dependencies.gain_lookup,
                                                          direct_event_dependencies.range_fit_lookup))

        return HitDirectEventDataProduct(event_outputs=processed_events, input_metadata=self.input_metadata)

    def process_pitch_angle_product(self) -> HitPitchAngleDataProduct:
        number_of_pitch_angle_bins = 8
        number_of_gyrophase_bins = 15
        dependencies = HITL3SectoredDependencies.fetch_dependencies(self.dependencies)
        mag_data = dependencies.mag_l1d_data
        hit_data = dependencies.data
        input_data_by_species = {"cno": hit_data.CNO, "helium4": hit_data.helium4, "hydrogen": hit_data.hydrogen,
                                 "iron": hit_data.iron, "NeMgSi": hit_data.NeMgSi}
        rebinned_flux_by_species = {"cno": np.full_like(hit_data.CNO, np.nan),
                                    "helium4": np.full_like(hit_data.helium4, np.nan),
                                    "hydrogen": np.full_like(hit_data.hydrogen, np.nan),
                                    "iron": np.full_like(hit_data.iron, np.nan),
                                    "NeMgSi": np.full_like(hit_data.NeMgSi, np.nan)}
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
        dec, dec_delta, inc, inc_delta = get_hit_bin_polar_coordinates()
        sector_unit_vectors = get_sector_unit_vectors(dec, inc)
        particle_unit_vectors = -sector_unit_vectors
        sector_areas = calculate_sector_areas(dec, dec_delta, inc_delta)
        pitch_angles, pitch_angle_deltas, gyrophases, gyrophase_delta = get_hit_bin_polar_coordinates(
            number_of_pitch_angle_bins, number_of_gyrophase_bins)
        averaged_mag_data = mag_data.rebin_to(hit_data.epoch, hit_data.epoch_delta)
        for time_index, average_mag_vector in enumerate(averaged_mag_data):
            mag_unit_vector = calculate_unit_vector(average_mag_vector)
            input_bin_pitch_angles = calculate_pitch_angle(particle_unit_vectors, mag_unit_vector)
            input_bin_gyrophases = calculate_gyrophase(particle_unit_vectors, mag_unit_vector)

            for species, flux in input_data_by_species.items():
                rebinned_flux_by_species[species][time_index] = rebin_by_pitch_angle_and_gyrophase(flux[time_index],
                                                                                                   input_bin_pitch_angles,
                                                                                                   input_bin_gyrophases,
                                                                                                   sector_areas,
                                                                                                   number_of_pitch_angle_bins,
                                                                                                   number_of_gyrophase_bins)
        data_product = HitPitchAngleDataProduct(hit_data.epoch, hit_data.epoch_delta, pitch_angles, pitch_angle_deltas,
                                                gyrophases,
                                                gyrophase_delta,
                                                rebinned_flux_by_species["hydrogen"],
                                                h_energy_center,
                                                h_energy_delta,
                                                rebinned_flux_by_species["helium4"],
                                                he4_energy_center, he4_energy_delta, rebinned_flux_by_species["cno"],
                                                cno_energy_center, cno_energy_delta,
                                                rebinned_flux_by_species["NeMgSi"], nemgsi_energy_center,
                                                nemgsi_energy_delta, rebinned_flux_by_species["iron"], fe_energy_center,
                                                fe_energy_delta)

        return data_product
